"""Adaptador del backend Responses de Codex (contrato no oficial)."""

from __future__ import annotations

import asyncio
import base64
import codecs
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import cast

import httpx

from oraculo.domain.attachments import AttachmentMode, AttachmentPayload
from oraculo.domain.errors import AIError
from oraculo.domain.models import AIToolCall, ChatMessage, MessageRole, SkillTurnContext
from oraculo.domain.ports import EMPTY_SKILL_CONTEXT, WorkspaceToolExecutor
from oraculo.infrastructure.codex_auth import (
    CLIENT_ID, AuthError, OAuthCredential, OAuthCredentialStore,
)
from oraculo.infrastructure.openai_compat import (
    OpenAICompatClient, _summary_data, _summary_instructions, _validated_summary,
)
from oraculo.infrastructure.provider_errors import (
    extract_provider_error_message,
    read_bounded_error_body,
)

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
TOKEN_URL = "https://auth.openai.com/oauth/token"
MAX_SSE_RESPONSE_BYTES = 256 * 1024
MAX_OUTPUT_TEXT_BYTES = 128 * 1024
MAX_TOOL_ARGUMENT_BYTES = 32 * 1024

_LOAD_SKILL_TOOL: dict[str, object] = {
    "type": "function", "name": "load_skill",
    "description": "Carga instrucciones de una skill disponible",
    "parameters": {
        "type": "object", "properties": {"name": {"type": "string"}},
        "required": ["name"], "additionalProperties": False,
    },
}

_WORKSPACE_TOOLS: tuple[dict[str, object], ...] = tuple(
    {
        "type": "function", "name": name, "description": description,
        "parameters": {
            "type": "object", "properties": properties,
            "required": required, "additionalProperties": False,
        },
    }
    for name, description, properties, required in (
        ("list_documents", "Lista documentos del espacio de trabajo", {}, []),
        ("read_document", "Lee un documento del espacio de trabajo",
         {"name": {"type": "string"}}, ["name"]),
        ("create_document", "Crea un documento nuevo sin sobrescribir",
         {"name": {"type": "string"}, "content": {"type": "string"},
          "kind": {"type": "string", "enum": ["final", "temp"]}},
         ["name", "content", "kind"]),
        ("propose_edit", "Propone una edición sin escribir hasta confirmación",
         {"name": {"type": "string"}, "content": {"type": "string"}},
         ["name", "content"]),
    )
)
_WORKSPACE_NAMES = frozenset(tool["name"] for tool in _WORKSPACE_TOOLS)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AIError("respuesta inesperada de la API")
    return cast(dict[str, object], value)


def _rejection_reason(status: int) -> str:
    """Motivo genérico y accionable para el operador, sin datos sensibles."""
    if status == 429:
        return "límite de uso de Codex alcanzado"
    if status in (401, 403):
        return "autorización de Codex rechazada"
    return "consulta Codex rechazada"


async def _bounded_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Acota bytes descomprimidos antes de decodificar o retener líneas SSE."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_SSE_RESPONSE_BYTES:
            raise AIError("tamaño de respuesta SSE excedido")
        pending += decoder.decode(chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            yield line.removesuffix("\r")
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending.removesuffix("\r")


async def _events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict[str, object]]]:
    name = ""
    data: list[str] = []
    async for line in lines:
        if not line:
            if name or data:
                if not data:
                    raise AIError("evento SSE inválido")
                if data == ["[DONE]"]:
                    name, data = "", []
                    continue
                try:
                    value: object = json.loads("\n".join(data))
                except ValueError:
                    raise AIError("evento SSE inválido") from None
                event_data = _mapping(value)
                event_name = event_data.get("type", name)
                if not isinstance(event_name, str) or not event_name:
                    raise AIError("evento SSE inválido")
                yield event_name, event_data
            name, data = "", []
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line.startswith(":"):
            continue
        else:
            raise AIError("evento SSE inválido")
    if name or data:
        raise AIError("evento SSE incompleto")


class CodexClient:
    """Cliente inyectado de una sola cuenta OAuth; cierra su HTTP con ``aclose``."""

    def __init__(
        self, *, store: OAuthCredentialStore, http_client: httpx.AsyncClient,
        system_prompt: str, model: str = "gpt-6-luna", timeout_s: float = 30.0,
    ) -> None:
        self._store = store
        self._http_client = http_client
        self._system_prompt = system_prompt
        self._model = model
        self._timeout_s = timeout_s
        self._refresh_lock = asyncio.Lock()

    async def _credential(self) -> OAuthCredential:
        # El lock protege también la relectura: otras tareas ven la rotación persistida.
        async with self._refresh_lock:
            try:
                credential = await self._store.load()
            except AuthError:
                raise AIError("credencial OAuth no disponible") from None
            if credential is None:
                raise AIError("credencial OAuth no disponible")
            if credential.expires_at > time.time() + 60:
                return credential
            try:
                response = await self._http_client.post(
                    TOKEN_URL, data={"grant_type": "refresh_token", "client_id": CLIENT_ID,
                                     "refresh_token": credential.refresh_token},
                    timeout=self._timeout_s, follow_redirects=False,
                )
                if response.status_code != 200:
                    raise AIError("renovación OAuth rechazada")
                value: object = response.json()
                data = _mapping(value)
                access, refresh, duration = (data.get("access_token"),
                                              data.get("refresh_token"), data.get("expires_in"))
                if (not isinstance(access, str) or not access
                        or not isinstance(refresh, str) or not refresh
                        or type(duration) is not int or duration <= 0):
                    raise AIError("respuesta OAuth inválida")
                rotated = OAuthCredential(access, refresh, int(time.time()) + duration,
                                          credential.account_id)
                await self._store.save(rotated)
                return rotated
            except (httpx.HTTPError, AuthError, ValueError, TypeError):
                raise AIError("renovación OAuth fallida") from None

    @staticmethod
    def _merge_attachments(
        input_items: list[dict[str, object]],
        attachments: Sequence[AttachmentPayload],
    ) -> None:
        """Añade los adjuntos al último mensaje de usuario.

        Codex (Responses API) admite `input_file` con `file_data` base64 para
        documentos y texto, e `input_image` para imágenes. El proveedor rechaza
        lo que no soporte y ese error se propaga al usuario.
        """
        blocks: list[dict[str, object]] = []
        for attachment in attachments:
            if attachment.mode is AttachmentMode.TEXT:
                blocks.append({
                    "type": "input_text",
                    "text": f"[Archivo adjunto: {attachment.name}]\n{attachment.text or ''}",
                })
            else:
                encoded = base64.b64encode(attachment.data or b"").decode("ascii")
                data_url = f"data:{attachment.mime_type};base64,{encoded}"
                if attachment.mime_type.startswith("image/"):
                    blocks.append({
                        "type": "input_image",
                        "image_url": data_url,
                    })
                else:
                    blocks.append({
                        "type": "input_file",
                        "filename": attachment.name,
                        "file_data": data_url,
                    })
        for index in range(len(input_items) - 1, -1, -1):
            if input_items[index].get("role") == "user":
                content = input_items[index]["content"]
                merged = list(content) if isinstance(content, list) else []
                input_items[index]["content"] = [*merged, *blocks]
                return
        input_items.append({"role": "user", "content": blocks})

    async def _response(
        self, payload: dict[str, object],
    ) -> tuple[str, tuple[AIToolCall, ...]]:
        credential = await self._credential()
        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "ChatGPT-Account-Id": credential.account_id,
            "originator": "oraculo", "User-Agent": "oraculo",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
        }
        text: list[str] = []
        text_bytes = 0
        argument_bytes = 0
        calls: list[AIToolCall] = []
        completed = False
        try:
            async with self._http_client.stream(
                "POST", CODEX_URL, json=payload, headers=headers,
                timeout=self._timeout_s, follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    body = await read_bounded_error_body(response)
                    raise AIError(
                        _rejection_reason(response.status_code),
                        user_message=extract_provider_error_message(
                            body, secrets=(credential.access_token,),
                        ),
                    )
                # El backend real responde 200 sin content-type; el SSE lo valida el parser.
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in ("", "text/event-stream"):
                    raise AIError("respuesta inesperada de la API")
                async for name, data in _events(_bounded_lines(response)):
                    if completed:
                        raise AIError("respuesta inesperada de la API")
                    if name == "response.output_text.delta":
                        delta = data.get("delta")
                        if not isinstance(delta, str):
                            raise AIError("respuesta inesperada de la API")
                        text_bytes += len(delta.encode("utf-8"))
                        if text_bytes > MAX_OUTPUT_TEXT_BYTES:
                            raise AIError("tamaño de texto de la API excedido")
                        text.append(delta)
                    elif name == "response.output_item.done":
                        item = _mapping(data.get("item"))
                        if item.get("type") == "function_call":
                            call_id, tool_name, arguments = (item.get("call_id"), item.get("name"),
                                                              item.get("arguments"))
                            if (not isinstance(call_id, str) or not call_id
                                    or any(call.tool_call_id == call_id for call in calls)
                                    or not isinstance(tool_name, str)
                                    or not isinstance(arguments, str)):
                                raise AIError("respuesta de tools inesperada de la API")
                            argument_bytes += len(arguments.encode("utf-8"))
                            if argument_bytes > MAX_TOOL_ARGUMENT_BYTES:
                                raise AIError("tamaño de argumentos de la API excedido")
                            calls.append(AIToolCall(call_id, tool_name, arguments))
                    elif name in ("response.completed", "response.done"):
                        status = _mapping(data.get("response", {})).get("status", "completed")
                        if status != "completed":
                            raise AIError("respuesta incompleta de la API")
                        completed = True
                    elif name in ("error", "response.failed", "response.incomplete"):
                        raise AIError(
                            "consulta Codex fallida",
                            user_message=extract_provider_error_message(
                                data, secrets=(credential.access_token,),
                            ),
                        )
        except httpx.HTTPError:
            raise AIError("fallo de red al contactar Codex") from None
        if not completed or (calls and text) or (not calls and not "".join(text).strip()):
            raise AIError("respuesta incompleta de la API")
        return "".join(text), tuple(calls)

    async def complete(
        self, messages: Sequence[ChatMessage], *,
        skills: SkillTurnContext = EMPTY_SKILL_CONTEXT,
        workspace_tools: WorkspaceToolExecutor | None = None,
        attachments: Sequence[AttachmentPayload] = (),
    ) -> str:
        instructions = "\n\n".join(
            [self._system_prompt]
            + ([skills.catalog_text] if skills.catalog_text else [])
            + [OpenAICompatClient._skill_block(skill) for skill in skills.explicit_skills]
        )
        context_chars = len(skills.catalog_text) + sum(
            OpenAICompatClient._skill_block_size(skill) for skill in skills.explicit_skills
        )
        if context_chars > skills.max_context_chars:
            raise AIError("presupuesto de contexto de skills excedido")
        input_items: list[dict[str, object]] = [
            {"role": message.role.value, "content": [{
                "type": "input_text" if message.role is MessageRole.USER else "output_text",
                "text": message.content,
            }]} for message in messages
        ]
        if attachments:
            self._merge_attachments(input_items, attachments)
        payload: dict[str, object] = {
            "model": self._model, "instructions": instructions,
            "input": input_items, "store": False, "stream": True,
        }
        skill_tools_enabled = skills.autoload_enabled and bool(skills.loadable_skills)
        tools_enabled = skill_tools_enabled or workspace_tools is not None
        if tools_enabled:
            payload["tools"] = ([_LOAD_SKILL_TOOL] if skill_tools_enabled else []) + (
                list(_WORKSPACE_TOOLS) if workspace_tools is not None else []
            )
            payload["tool_choice"] = "auto"
        allowlist = {skill.name: skill for skill in skills.loadable_skills}
        loaded_names = {skill.name for skill in skills.explicit_skills}
        rounds = 0
        while True:
            text, calls = await self._response(payload)
            if not calls:
                return text
            if not tools_enabled:
                raise AIError("respuesta de tools inesperada de la API")
            if rounds >= skills.max_rounds:
                raise AIError("máximo de rondas de skills excedido")
            rounds += 1
            for call in calls:
                if call.name == "load_skill" and skill_tools_enabled:
                    result, loaded = OpenAICompatClient._resolve_tool_call(call, allowlist, loaded_names)
                elif workspace_tools is not None and call.name in _WORKSPACE_NAMES:
                    result = await workspace_tools.execute(call.name, call.arguments_json)
                    loaded = None
                else:
                    result, loaded = {"status": "unsupported_tool"}, None
                if loaded is not None:
                    context_chars += OpenAICompatClient._skill_block_size(loaded)
                    if context_chars > skills.max_context_chars:
                        raise AIError("presupuesto de contexto de skills excedido")
                    loaded_names.add(loaded.name)
                input_items.extend((
                    {"type": "function_call", "call_id": call.tool_call_id,
                     "name": call.name, "arguments": call.arguments_json},
                    {"type": "function_call_output", "call_id": call.tool_call_id,
                     "output": OpenAICompatClient._json(result)},
                ))
            payload["input"] = input_items

    async def summarize(
        self, previous_summary: str | None, messages: Sequence[ChatMessage], *,
        max_chars: int,
    ) -> str:
        """Genera un resumen con la cuenta y el modelo configurados, sin tools."""
        if max_chars < 1:
            raise AIError("límite de resumen inválido")
        payload: dict[str, object] = {
            "model": self._model,
            "instructions": _summary_instructions(max_chars),
            "input": [{"role": "user", "content": [{
                "type": "input_text", "text": _summary_data(previous_summary, messages),
            }]}],
            "store": False,
            "stream": True,
        }
        try:
            text, calls = await self._response(payload)
        except AIError:
            raise AIError("fallo al generar el resumen") from None
        if calls:
            raise AIError("respuesta de resumen inesperada de la API")
        return _validated_summary(text, max_chars)

    async def aclose(self) -> None:
        await self._http_client.aclose()
