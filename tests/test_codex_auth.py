"""Pruebas aisladas del login Codex; nunca contactan al proveedor real."""

import asyncio
import base64
import hashlib
import json
import os
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from oraculo.infrastructure.codex_auth import (
    AuthError,
    CodexBrowserLogin,
    OAuthCredential,
    OAuthCredentialStore,
    _callback_code,
)


def credential() -> OAuthCredential:
    return OAuthCredential("access-SENTINEL", "refresh-SENTINEL", 1900000000, "account-1")


def callback(state: str, *, code: str = "code-SENTINEL", route: str = "/auth/callback") -> bytes:
    return f"GET {route}?code={code}&state={state} HTTP/1.1\r\nHost: localhost:1455\r\n\r\n".encode()


async def request(raw: bytes) -> int:
    reader, writer = await asyncio.open_connection("127.0.0.1", 1455)
    writer.write(raw)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return int(response.split(b" ", 2)[1])


def token_response() -> dict[str, object]:
    claim = {"https://api.openai.com/auth": {"chatgpt_account_id": "account-1"}}
    payload = base64.urlsafe_b64encode(json.dumps(claim).encode()).decode().rstrip("=")
    return {"access_token": f"header.{payload}.signature", "refresh_token": "refresh-SENTINEL", "expires_in": 3600}


@pytest.mark.asyncio
async def test_store_roundtrip_permissions_and_redacted_repr(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    await store.save(credential())
    assert await store.load() == credential()
    assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert "SENTINEL" not in repr(credential())
    await store.save(OAuthCredential("next-access", "next-refresh", 1900000001, "account-1"))
    assert (await store.load()).refresh_token == "next-refresh"


@pytest.mark.asyncio
async def test_store_rejects_workspace_symlink_and_malformed_data(tmp_path: Path) -> None:
    from oraculo.infrastructure import codex_auth

    with pytest.raises(AuthError):
        OAuthCredentialStore(Path(codex_auth.__file__).parents[3] / "auth.json")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "auth.json"
    path.symlink_to(tmp_path / "target")
    store = OAuthCredentialStore(path)
    with pytest.raises(AuthError):
        await store.save(credential())
    with pytest.raises(AuthError):
        await store.load()
    path.unlink()
    path.write_text('{"version":1,"access_token":"SENTINEL"}')
    os.chmod(path, 0o600)
    with pytest.raises(AuthError) as exc:
        await store.load()
    assert "SENTINEL" not in str(exc.value)


@pytest.mark.asyncio
async def test_login_pkce_callback_exchange_and_persist(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    urls: list[str] = []
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client, issuer="https://issuer.example", timeout_s=2)
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://issuer.example/oauth/token").mock(return_value=httpx.Response(200, json=token_response()))
            async def notify(url: str) -> None:
                urls.append(url)
                query = parse_qs(urlsplit(url).query)
                assert urlsplit(url).path == "/oauth/authorize"
                assert query["redirect_uri"] == ["http://localhost:1455/auth/callback"]
                assert query["code_challenge_method"] == ["S256"]
                assert query["scope"] == ["openid profile email offline_access"]
                state = query["state"][0]
                assert await request(callback(state, route="/bad")) == 404
                assert await request(callback("wrong")) == 400
                assert await request(callback(state, route="/auth/callback&x")) == 404
                assert await request(callback(state)) == 200

            result = await login.login(notify)
            sent = parse_qs(route.calls[0].request.content.decode())
            verifier = sent["code_verifier"][0]
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
            assert parse_qs(urlsplit(urls[0]).query)["code_challenge"] == [challenge]
            assert sent["code"] == ["code-SENTINEL"]
            assert sent["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
            assert sent["redirect_uri"] == ["http://localhost:1455/auth/callback"]
            assert result == await store.load()
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", 1455)


@pytest.mark.asyncio
async def test_callback_acepta_scope_extra_sin_relajar_validacion_de_state(tmp_path: Path) -> None:
    """El proveedor añade scope junto al code y state; usar solo datos ficticios."""
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client, issuer="https://issuer.example", timeout_s=2)
        with respx.mock:
            route = respx.post("https://issuer.example/oauth/token").mock(
                return_value=httpx.Response(200, json=token_response()))

            async def notify(url: str) -> None:
                state = parse_qs(urlsplit(url).query)["state"][0]
                assert await request(callback("other-state", code="fake-code&scope=openid")) == 400
                assert await request(callback(state, code="fake-code&scope=openid+profile+email+offline_access")) == 200

            await login.login(notify)
            assert parse_qs(route.calls.last.request.content.decode())["code"] == ["fake-code"]


def test_validacion_callback_con_scope_extra_y_proteccion_csrf() -> None:
    query = "code=fake-code&scope=openid+profile+email+offline_access&state=expected"
    assert _callback_code(query, "expected") == "fake-code"
    assert _callback_code(query, "wrong") is None
    assert _callback_code(query + "&state=expected", "expected") is None
    assert _callback_code("code=fake-code&state=expected&code=second", "expected") is None


@pytest.mark.asyncio
async def test_login_timeout_cancellation_and_no_secret_leak(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client, issuer="https://issuer.example", timeout_s=0.02)
        with pytest.raises(AuthError, match="tiempo"):
            await login.login(lambda url: None)
        assert await store.load() is None
        task = asyncio.create_task(CodexBrowserLogin(store=store, http_client=client, issuer="https://issuer.example").login(lambda url: None))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", 1455)


@pytest.mark.asyncio
async def test_login_rejects_bad_token_without_storing(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client, issuer="https://issuer.example", timeout_s=2)
        with respx.mock:
            respx.post("https://issuer.example/oauth/token").mock(return_value=httpx.Response(400, json={"error": "refresh-SENTINEL"}))
            async def notify(url: str) -> None:
                await request(callback(parse_qs(urlsplit(url).query)["state"][0]))

            with pytest.raises(AuthError) as exc:
                await login.login(notify)
            assert "SENTINEL" not in str(exc.value)
            assert await store.load() is None


@pytest.mark.asyncio
async def test_occupied_callback_port_fails_before_notify(tmp_path: Path) -> None:
    notified: list[str] = []
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=OAuthCredentialStore(tmp_path / "private" / "auth.json"),
                                  http_client=client, issuer="https://issuer.example")
        with socket.socket() as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", 1455))
            occupied.listen()
            with pytest.raises(AuthError, match="puerto"):
                await login.login(notified.append)
    assert notified == []


@pytest.mark.asyncio
async def test_store_failed_replace_keeps_previous_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    await store.save(credential())

    def fail_replace(source: str, target: Path) -> None:
        raise OSError("refresh-SENTINEL")

    monkeypatch.setattr("oraculo.infrastructure.codex_auth.os.replace", fail_replace)
    with pytest.raises(AuthError) as exc:
        await store.save(OAuthCredential("next", "next", 1900000001, "account-1"))
    assert "SENTINEL" not in str(exc.value)
    assert await store.load() == credential()
    assert list(store.path.parent.iterdir()) == [store.path]


@pytest.mark.asyncio
async def test_login_rejects_redirect(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client,
                                  issuer="https://issuer.example", timeout_s=2)
        with respx.mock:
            route = respx.post("https://issuer.example/oauth/token").mock(
                return_value=httpx.Response(302, headers={"Location": "https://other.example/token"}))

            async def notify(url: str) -> None:
                state = parse_qs(urlsplit(url).query)["state"][0]
                assert await request(callback(state, code="x", route="/other")) == 404
                assert await request(callback(state, code="x&state=duplicate")) == 400
                assert await request(callback(state, code="x")) == 200

            with pytest.raises(AuthError):
                await login.login(notify)
            assert route.call_count == 1
            assert await store.load() is None


@pytest.mark.asyncio
async def test_login_rejects_missing_account_claim(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client,
                                  issuer="https://issuer.example", timeout_s=2)
        with respx.mock:
            respx.post("https://issuer.example/oauth/token").mock(return_value=httpx.Response(200, json={
                "access_token": "not-a-jwt-SENTINEL", "refresh_token": "refresh-SENTINEL", "expires_in": 3600,
            }))

            async def notify(url: str) -> None:
                assert await request(callback(parse_qs(urlsplit(url).query)["state"][0])) == 200

            with pytest.raises(AuthError) as exc:
                await login.login(notify)
            assert "SENTINEL" not in str(exc.value)
            assert await store.load() is None
