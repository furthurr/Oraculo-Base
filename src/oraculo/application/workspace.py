"""Capacidades documentales efímeras, autorizadas por mensaje y ámbito."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from oraculo.domain.models import ChatKind, IncomingMessage
from oraculo.domain.ports import DocumentStore
from oraculo.domain.workspace import DocumentRef, DocumentVersion, WorkspaceError, WorkspaceScope


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _ref(name: object) -> DocumentRef:
    if not isinstance(name, str):
        raise WorkspaceError("invalid_reference")
    temporary = name.startswith("temp/")
    return DocumentRef(name[5:] if temporary else name, temporary)


def _label(ref: DocumentRef) -> str:
    return ("temp/" if ref.temporary else "") + ref.name


@dataclass(frozen=True)
class _Proposal:
    scope: WorkspaceScope
    ref: DocumentRef
    content: str
    version: DocumentVersion
    code: str
    expires: datetime
    delivered: bool = False


class WorkspaceService:
    def __init__(
        self,
        store: DocumentStore,
        allowed_user_ids: frozenset[int],
        allowed_chat_ids: frozenset[int],
        workspace_group_ids: frozenset[int],
    ) -> None:
        self._store = store
        self._users = frozenset(allowed_user_ids)
        self._chats = frozenset(allowed_chat_ids)
        self._workspace_group_ids = frozenset(workspace_group_ids)
        self._locks: dict[WorkspaceScope, asyncio.Lock] = {}
        self._pending: dict[tuple[int, int], _Proposal] = {}

    def _scope(self, message: IncomingMessage) -> WorkspaceScope | None:
        if message.chat_type is ChatKind.PRIVATE and message.user_id in self._users:
            return WorkspaceScope(ChatKind.PRIVATE, message.user_id)
        if (message.chat_type is ChatKind.GROUP and message.chat_id in self._chats
                and message.chat_id in self._workspace_group_ids):
            return WorkspaceScope(ChatKind.GROUP, message.chat_id)
        return None

    def _lock(self, scope: WorkspaceScope) -> asyncio.Lock:
        return self._locks.setdefault(scope, asyncio.Lock())

    def workspace_for(self, message: IncomingMessage) -> WorkspaceTurn | None:
        scope = self._scope(message)
        if scope is None:
            return None
        if message.chat_type is ChatKind.GROUP and not self._explicit_request(message.text or ""):
            return None
        return WorkspaceTurn(self, message, scope)

    @staticmethod
    def _explicit_request(text: str) -> bool:
        """Sólo el texto actual puede autorizar tools; ni historial ni citas."""
        action = re.search(
            r"\b(?:list[aoar]*|mu[eé]strame|ens[eé][ñn]ame|dame|cu[aá]les\s+son|"
            r"qu[eé]\s+(?:documentos|archivos|hay|dice)|"
            r"lee|leer|abre|abrir|consulta|busca|crea|crear|genera|generar|"
            r"guarda|guardar|escribe|escribir|edita|editar|modifica|modificar|"
            r"actualiza|actualizar|reemplaza|reemplazar|resume|resumir|"
            r"show|list|read|open|create|write|edit|update|save)\b",
            text, re.IGNORECASE,
        )
        target = re.search(r"\b(?:documentos?|archivos?|ficheros?|workspace|"
                           r"docs?|files?|[\w-]+\.(?:md|txt|pdf))\b", text, re.IGNORECASE)
        return action is not None and target is not None

    @staticmethod
    def _write_request(text: str) -> bool:
        """Un turno grupal de lectura nunca autoriza una escritura inferida del archivo."""
        return re.search(
            r"\b(?:crea|crear|genera|generar|guarda|guardar|escribe|escribir|"
            r"edita|editar|modifica|modificar|actualiza|actualizar|"
            r"reemplaza|reemplazar|create|write|edit|update|save)\b",
            text, re.IGNORECASE,
        ) is not None

    def preview_delivery(self, message: IncomingMessage, reply: str, sent: bool) -> None:
        """Publica una propuesta sólo después de entregar íntegra su vista previa."""
        key = (message.chat_id, message.user_id)
        if not sent:
            self._pending.pop(key, None)
            return
        proposal = self._pending.get(key)
        if proposal is not None and proposal.scope == self._scope(message) and proposal.code in reply:
            self._pending[key] = _Proposal(
                proposal.scope, proposal.ref, proposal.content, proposal.version,
                proposal.code, proposal.expires, delivered=True,
            )

    @staticmethod
    def _attachment_intent(text: str) -> bool:
        action = r"(?:env[ií]ame|adjunta)"
        if re.search(r"\b(?:no|nunca|sin|evita|tampoco)\b.{0,32}\b" + action + r"\b", text, re.IGNORECASE):
            return False
        if re.search(r"\b" + action + r"\b.{0,40}\b(?:no|nunca)\b", text, re.IGNORECASE):
            return False
        return re.search(r"\b" + action + r"\b", text, re.IGNORECASE) is not None

    @staticmethod
    def _attachment_ref(text: str) -> DocumentRef | None:
        # Nunca elegir entre varios nombres, aunque sólo uno tenga una extensión permitida.
        candidates = [token.strip(",;!?") for token in text.split()
                      if "." in token or "/" in token or "\\" in token]
        if len(candidates) != 1:
            return None
        try:
            return _ref(candidates[0])
        except WorkspaceError:
            return None

    def attachment_clarification(self, message: IncomingMessage) -> str | None:
        """Respuesta local tras una solicitud de adjunto no entregada; sin I/O documental."""
        if not self._attachment_intent(message.text or ""):
            return None
        if self._scope(message) is None:
            return "No puedo enviar documentos de este espacio."
        return "Indica un único nombre de archivo .md o .txt (puedes incluir temp/)."

    async def confirm(self, message: IncomingMessage) -> str | None:
        text = (message.text or "").strip()
        match = re.fullmatch(r"confirma\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        cancel = re.fullmatch(r"cancela", text, re.IGNORECASE)
        if not match and not cancel:
            return None
        scope = self._scope(message)
        if scope is None:
            return None
        key = (message.chat_id, message.user_id)
        async with self._lock(scope):
            proposal = self._pending.get(key)
            if proposal is None or proposal.scope != scope or not proposal.delivered:
                return "No hay propuesta vigente; no se aplicó ningún cambio."
            if cancel:
                del self._pending[key]
                return "Propuesta cancelada; no se aplicó ningún cambio."
            if datetime.now(timezone.utc) >= proposal.expires:
                del self._pending[key]
                return "La propuesta venció; no se aplicó ningún cambio."
            if not secrets.compare_digest(match.group(1), proposal.code):
                return "Código incorrecto; no se aplicó ningún cambio."
            del self._pending[key]
            try:
                await self._store.replace_if_version(scope, proposal.ref, proposal.content, proposal.version)
            except WorkspaceError as exc:
                if exc.reason == "conflict":
                    return "El documento cambió; solicita una nueva vista previa. No se aplicó el cambio."
                if exc.reason == "durability_unknown":
                    return "El documento pudo actualizarse, pero no se pudo confirmar su guardado. Comprueba el archivo antes de reintentar."
                return f"No se aplicó el cambio ({exc.reason})."
            return f"Documento {_label(proposal.ref)} actualizado."

    async def attachment(self, message: IncomingMessage) -> tuple[str, bytes] | None:
        scope = self._scope(message)
        text = (message.text or "").strip()
        if scope is None or not self._attachment_intent(text):
            return None
        # Sólo una referencia explícita; nunca adivinar a partir de texto citado.
        ref = self._attachment_ref(text)
        if ref is None:
            return None
        try:
            return _label(ref), await self._store.attachment(scope, ref)
        except WorkspaceError:
            return None

    def reset(self, message: IncomingMessage) -> None:
        scope = self._scope(message)
        if scope is not None:
            for key, proposal in tuple(self._pending.items()):
                if proposal.scope == scope:
                    del self._pending[key]


class WorkspaceTurn:
    """Ejecutor por turno; no admite ámbito suministrado por el modelo."""

    def __init__(self, service: WorkspaceService, message: IncomingMessage, scope: WorkspaceScope) -> None:
        self._service = service
        self._message = message
        self._scope = scope
        self._count = 0
        self.notices: list[str] = []
        self.read_ref: str | None = None
        self.long_read: bool = False

    async def execute(self, name: str, arguments_json: str) -> dict[str, object]:
        if self._service._scope(self._message) != self._scope:
            return self._error("access_denied")
        if self._count >= 5:
            return self._error("operation_limit")
        self._count += 1
        keys = {
            "list_documents": frozenset(),
            "read_document": frozenset({"name"}),
            "create_document": frozenset({"name", "content", "kind"}),
            "propose_edit": frozenset({"name", "content"}),
        }
        if name not in keys:
            return self._error("unknown_tool")
        if (self._scope.kind is ChatKind.GROUP and name in {"create_document", "propose_edit"}
                and not self._service._write_request(self._message.text or "")):
            return self._error("access_denied")
        try:
            args = json.loads(arguments_json, object_pairs_hook=_unique_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
            if not isinstance(args, dict) or not set(args) <= keys[name]:
                return self._error("invalid_arguments")
            if name == "list_documents":
                if args:
                    return self._error("invalid_arguments")
                refs = await self._service._store.list(self._scope)
                labels = [_label(ref) for ref in refs]
                visibility = " Visible para el grupo." if self._scope.kind is ChatKind.GROUP else ""
                self.notices.append(
                    ("Documentos en el espacio autorizado: " + ", ".join(labels)
                     if labels else "No hay documentos en el espacio autorizado.") + visibility
                )
                return {"ok": True, "documents": labels}
            ref = _ref(args.get("name"))
            label = _label(ref)
            if name == "read_document":
                if set(args) != {"name"}:
                    return self._error("invalid_arguments")
                requested = (self._message.text or "").lower()
                if label.lower() not in requested and ref.name.lower() not in requested:
                    if re.search(r"[a-z0-9][a-z0-9_.-]*\.(?:md|txt)\b", requested):
                        return self._error("access_denied")
                    available = await self._service._store.list(self._scope)
                    if len(available) != 1:
                        return self._error("needs_clarification")
                    if available[0] != ref:
                        return self._error("access_denied")
                content, _ = await self._service._store.read(self._scope, ref)
                self.read_ref = label
                show_content = re.search(
                    r"\b(?:lee|leer|abre|abrir|mu[eé]strame|dame|read|open)\b",
                    self._message.text or "", re.IGNORECASE,
                ) is not None
                if len(content) > 1900:
                    self.long_read = True
                    if self._scope.kind is ChatKind.PRIVATE:
                        self.notices.append(f"El documento {label} puede ocupar varios mensajes. Si lo prefieres como archivo, pide: envíame {label}.")
                    else:
                        self.notices.append(f"El documento {label} no cabe en un mensaje del grupo; acota la consulta a un fragmento. Cualquier contenido publicado será visible para el grupo.")
                    if show_content and self._scope.kind is ChatKind.PRIVATE:
                        self.notices.append(f"Contenido de {label}:\n{content}")
                    return {"ok": True, "name": label, "content": content}
                self.notices.append(f"Documento leído: {label}." + (" Visible para el grupo." if self._scope.kind is ChatKind.GROUP else ""))
                if show_content:
                    self.notices.append(f"Contenido de {label}:\n{content}")
                return {"ok": True, "content": content}
            content = args.get("content")
            if not isinstance(content, str):
                return self._error("invalid_arguments")
            if name == "create_document":
                kind = args.get("kind", "temp")
                if kind not in ("temp", "final") or (ref.temporary and kind == "final"):
                    return self._error("invalid_arguments")
                request = (self._message.text or "").lower()
                if re.search(r"\b(?:temporal|temporales|borrador|borradores|provisional)\b", request):
                    kind = "temp"
                elif re.search(r"\b(?:final|definitiv[oa]|conserva|guardar|guarda|guárdalo|gu[aá]rdame)\b", request):
                    kind = "final"
                ref = DocumentRef(ref.name, temporary=kind == "temp")
                label = _label(ref)
                async with self._service._lock(self._scope):
                    try:
                        await self._service._store.create(self._scope, ref, content)
                    except WorkspaceError as exc:
                        if exc.reason == "durability_unknown":
                            self.notices.append(
                                f"El documento {label} pudo crearse, pero no se pudo confirmar su guardado. "
                                "Comprueba el archivo antes de reintentar."
                            )
                            return {"error": "durability_unknown"}
                        raise
                self.notices.append(f"Documento creado: {label}." + (" Visible para el grupo." if self._scope.kind is ChatKind.GROUP else ""))
                return {"ok": True, "name": label}
            if set(args) != {"name", "content"}:
                return self._error("invalid_arguments")
            async with self._service._lock(self._scope):
                previous, version = await self._service._store.read(self._scope, ref)
                if len(content) > 3000 or len(content.encode("utf-8")) > 8192:
                    return self._error("quota")
                code = secrets.token_urlsafe(18)
                self._service._pending[(self._message.chat_id, self._message.user_id)] = _Proposal(
                    self._scope, ref, content, version, code,
                    datetime.now(timezone.utc) + timedelta(minutes=10),
                )
            self.notices.append(
                f"Vista previa completa de {label} (sin aplicar):\nAnterior:\n{previous}\nPropuesto:\n{content}\n"
                f"Para aplicar, responde: confirma {code}." +
                (" Visible para el grupo." if self._scope.kind is ChatKind.GROUP else "")
            )
            return {"ok": True, "code": code, "name": label}
        except (ValueError, TypeError, UnicodeEncodeError):
            return self._error("invalid_arguments")
        except WorkspaceError as exc:
            return self._error(exc.reason)

    def _error(self, reason: str) -> dict[str, object]:
        self.notices.append(f"Operación documental no realizada ({reason}).")
        return {"error": reason}
