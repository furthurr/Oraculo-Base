"""Credenciales locales y login por navegador para Codex (contrato no oficial).

Solo el llamador CLI debe pasar ``notify``; nunca enviar la URL a Telegram.
El cliente HTTP es inyectado y su ciclo de vida pertenece al llamador.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import stat
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # ID público observado en Pi/OpenCode.
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
_WORKSPACE = Path(__file__).resolve().parents[3]


class AuthError(Exception):
    """Fallo de autenticación o almacenaje, sin detalles sensibles."""


@dataclass(frozen=True, repr=False)
class OAuthCredential:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: int
    account_id: str = field(repr=False)

    def __post_init__(self) -> None:
        if (not all(isinstance(v, str) and v for v in
                    (self.access_token, self.refresh_token, self.account_id))
                or type(self.expires_at) is not int or self.expires_at <= 0):
            raise AuthError("credencial OAuth inválida")

    def __repr__(self) -> str:
        return "OAuthCredential(<redacted>)"


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


class OAuthCredentialStore:
    """Almacén de una cuenta; ``path`` absoluto fuera del workspace."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if (not self.path.is_absolute() or self.path == _WORKSPACE
                or self.path.is_relative_to(_WORKSPACE)
                or self.path.resolve().is_relative_to(_WORKSPACE)):
            raise AuthError("ruta de credenciales inválida")

    def _directory(self, *, create: bool) -> bool:
        parent = self.path.parent
        if parent.is_symlink() or parent.resolve().is_relative_to(_WORKSPACE):
            raise AuthError("directorio de credenciales inválido")
        if not parent.exists():
            if not create:
                return False
            # No crear cadenas de directorios ni usar enlaces en la ubicación final.
            parent.mkdir(mode=0o700)
        info = parent.stat()
        mode = info.st_mode
        if not stat.S_ISDIR(mode) or mode & 0o077 or info.st_uid != os.getuid():
            raise AuthError("permisos del directorio de credenciales inseguros")
        return True

    def _check_file(self) -> bool:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise AuthError("archivo de credenciales inseguro")
        return True

    def _load(self) -> OAuthCredential | None:
        if not self._directory(create=False) or not self._check_file():
            return None
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise AuthError("archivo de credenciales inseguro")
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise AuthError("credencial OAuth inválida")
        try:
            value: object = json.loads(raw, object_pairs_hook=_object)
            if not isinstance(value, dict) or set(value) != {
                "version", "access_token", "refresh_token", "expires_at", "account_id"
            } or type(value["version"]) is not int or value["version"] != 1:
                raise AuthError("credencial OAuth inválida")
            return OAuthCredential(value["access_token"], value["refresh_token"],
                                   value["expires_at"], value["account_id"])
        except (ValueError, TypeError, UnicodeError):
            raise AuthError("credencial OAuth inválida") from None

    async def load(self) -> OAuthCredential | None:
        try:
            return await asyncio.to_thread(self._load)
        except OSError:
            raise AuthError("no se pudo leer la credencial OAuth") from None

    def available_for_startup(self) -> bool:
        """Comprueba la credencial antes de abrir red/polling (en composition root)."""
        try:
            return self._load() is not None
        except OSError:
            raise AuthError("no se pudo leer la credencial OAuth") from None

    def _save(self, credential: OAuthCredential) -> None:
        self._directory(create=True)
        self._check_file()
        content = json.dumps({"version": 1, "access_token": credential.access_token,
                              "refresh_token": credential.refresh_token,
                              "expires_at": credential.expires_at,
                              "account_id": credential.account_id}).encode()
        fd, name = tempfile.mkstemp(prefix=".oauth-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_file()
            os.replace(name, self.path)
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    async def save(self, credential: OAuthCredential) -> None:
        if not isinstance(credential, OAuthCredential):
            raise AuthError("credencial OAuth inválida")
        try:
            await asyncio.to_thread(self._save, credential)
        except OSError:
            raise AuthError("no se pudo guardar la credencial OAuth") from None


def _account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("not JWT")
        payload: object = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        if not isinstance(payload, dict):
            raise ValueError("not object")
        claims = payload.get("https://api.openai.com/auth")
        if not isinstance(claims, dict):
            raise ValueError("missing claim")
        account = claims.get("chatgpt_account_id")
        if not isinstance(account, str) or not account:
            raise ValueError("missing account")
        return account
    except (ValueError, TypeError, UnicodeError) as exc:
        raise AuthError("respuesta OAuth inválida") from None


def _callback_code(query: str, state: str) -> str | None:
    """Acepta parámetros informativos adicionales sin relajar PKCE/state.

    No acepta duplicados de code/state ni callbacks que incluyan error.
    """
    args = parse_qs(query, keep_blank_values=True, strict_parsing=True)
    codes = args.get("code", [])
    states = args.get("state", [])
    if ("error" in args or len(codes) != 1 or not codes[0]
            or len(states) != 1 or not secrets.compare_digest(states[0], state)):
        return None
    return codes[0]


class CodexBrowserLogin:
    """Login único por instancia; listener efímero restringido a loopback."""

    def __init__(self, *, store: OAuthCredentialStore, http_client: httpx.AsyncClient,
                 issuer: str = "https://auth.openai.com", timeout_s: float = 300) -> None:
        parsed = urlsplit(issuer)
        if (parsed.scheme not in ("https", "http") or not parsed.hostname
                or parsed.username or parsed.password or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment or
                (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"))):
            raise AuthError("emisor OAuth inválido")
        if not 0 < timeout_s <= 300:
            raise AuthError("tiempo de login inválido")
        self.store = store
        self.http_client = http_client
        self.issuer = issuer.rstrip("/")
        self.timeout_s = timeout_s
        self._active = False

    async def login(self, notify: Callable[[str], None | Awaitable[None]]) -> OAuthCredential:
        if self._active:
            raise AuthError("ya hay un login en curso")
        self._active = True
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self._login(notify)
        except TimeoutError:
            raise AuthError("tiempo de login agotado") from None
        finally:
            self._active = False

    async def _login(self, notify: Callable[[str], None | Awaitable[None]]) -> OAuthCredential:
        verifier = secrets.token_urlsafe(32)
        state = secrets.token_urlsafe(32)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        loop = asyncio.get_running_loop()
        code_future: asyncio.Future[str] = loop.create_future()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            status = 400
            try:
                raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
                if len(raw) > 8192:
                    raise ValueError("large request")
                lines = raw.decode("ascii").split("\r\n")
                parts = lines[0].split(" ")
                if len(parts) != 3 or parts[0] != "GET" or parts[2] != "HTTP/1.1":
                    raise ValueError("invalid method")
                host = [line[5:].strip() for line in lines[1:] if line.lower().startswith("host:")]
                if host != ["localhost:1455"]:
                    raise ValueError("invalid host")
                url = urlsplit(parts[1])
                if url.path != "/auth/callback":
                    status = 404
                elif url.scheme or url.netloc or url.fragment or code_future.done():
                    raise ValueError("invalid callback")
                else:
                    code = _callback_code(url.query, state)
                    if code is not None:
                        status = 200
                        code_future.set_result(code)
            except (ValueError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
                pass
            try:
                message = b"Authentication complete. Close this page." if status == 200 else b"Invalid callback."
                writer.write(b"HTTP/1.1 " + str(status).encode() + b" Response\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: " + str(len(message)).encode() + b"\r\n\r\n" + message)
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        try:
            server = await asyncio.start_server(handle, "127.0.0.1", 1455, limit=8192)
        except OSError:
            raise AuthError("puerto de callback OAuth no disponible") from None
        try:
            params = {"response_type": "code", "client_id": CLIENT_ID,
                      "redirect_uri": REDIRECT_URI, "scope": SCOPE,
                      "code_challenge": challenge, "code_challenge_method": "S256",
                      "state": state, "id_token_add_organizations": "true",
                      "codex_cli_simplified_flow": "true", "originator": "oraculo"}
            result = notify(f"{self.issuer}/oauth/authorize?{urlencode(params)}")
            if asyncio.iscoroutine(result):
                await result
            code = await code_future
            try:
                response = await self.http_client.post(
                    f"{self.issuer}/oauth/token",
                    data={"grant_type": "authorization_code", "client_id": CLIENT_ID,
                          "code": code, "code_verifier": verifier, "redirect_uri": REDIRECT_URI},
                    timeout=min(30, self.timeout_s), follow_redirects=False,
                )
            except httpx.HTTPError:
                raise AuthError("fallo de red en OAuth") from None
            if response.status_code != 200:
                raise AuthError("intercambio OAuth rechazado")
            try:
                data: object = response.json()
                if not isinstance(data, dict):
                    raise ValueError("invalid token")
                access, refresh, duration = data.get("access_token"), data.get("refresh_token"), data.get("expires_in")
                if (not isinstance(access, str) or not access or not isinstance(refresh, str)
                        or not refresh or type(duration) is not int or duration <= 0):
                    raise ValueError("invalid token")
                credential = OAuthCredential(access, refresh, int(time.time()) + duration, _account_id(access))
            except (ValueError, TypeError):
                raise AuthError("respuesta OAuth inválida") from None
            await self.store.save(credential)
            return credential
        finally:
            server.close()
            await server.wait_closed()
