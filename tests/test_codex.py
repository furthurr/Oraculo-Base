"""Contrato Codex simulado; no se usan credenciales ni red reales."""

import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from oraculo.domain.errors import AIError
from oraculo.domain.models import ChatMessage, MessageRole, Skill, SkillTurnContext
from oraculo.domain.ports import AIClient, WorkspaceToolExecutor
from oraculo.infrastructure.codex import CodexClient
from oraculo.infrastructure.codex_auth import AuthError, OAuthCredential, OAuthCredentialStore

URL = "https://chatgpt.com/backend-api/codex/responses"
TOKEN_URL = "https://auth.openai.com/oauth/token"
MESSAGES = [ChatMessage(MessageRole.USER, "hola"), ChatMessage(MessageRole.ASSISTANT, "antes")]


def event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def sse(*events: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="".join(events))


def finished(text: str = "¡Hola!") -> httpx.Response:
    return sse(event("response.output_text.delta", {"delta": text}),
               event("response.completed", {"response": {"status": "completed"}}))


class ChunkedSSE(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.read_count = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.read_count += 1
            yield chunk


@pytest.mark.parametrize("kind", ["sse", "text", "arguments"])
async def test_rechaza_stream_excesivo_antes_de_seguir_leyendo(
    tmp_path: Path, kind: str,
) -> None:
    store = await setup(tmp_path)
    if kind == "sse":
        chunks = [b":" + b"x" * 140_000 + b"\n", b":" + b"x" * 140_000 + b"\n"]
    elif kind == "text":
        chunks = [event("response.output_text.delta", {"delta": "x" * 70_000}).encode(),
                  event("response.output_text.delta", {"delta": "x" * 70_000}).encode()]
    else:
        chunks = [tool_event(call_id="one", arguments='{"name":"' + "x" * 17_000 + '"}').encode(),
                  tool_event(call_id="two", arguments='{"name":"' + "x" * 17_000 + '"}').encode()]
    body = ChunkedSSE(chunks + [b"SECRET_PROVIDER_DATA"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(AIError, match="tamaño") as exc:
            await client(store, http).complete(MESSAGES, skills=context())
    assert body.read_count < len(body.chunks)
    assert exc.value.__cause__ is None
    assert "SECRET_PROVIDER_DATA" not in str(exc.value)


@respx.mock
async def test_sse_sin_campo_event_usa_type_del_json(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    body = '\n\n'.join([
        'data: {"type":"response.output_text.delta","delta":"hola"}',
        'data: {"type":"response.completed","response":{"status":"completed"}}',
    ]) + '\n\n'
    respx.post(URL).mock(return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body))
    async with httpx.AsyncClient() as http:
        assert await client(store, http).complete(MESSAGES) == "hola"


@respx.mock
async def test_sse_ignora_marcador_done_tras_completar(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    body = event("response.output_text.delta", {"delta": "hola"}) + event(
        "response.completed", {"response": {"status": "completed"}}
    ) + "data: [DONE]\n\n"
    respx.post(URL).mock(return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body))
    async with httpx.AsyncClient() as http:
        assert await client(store, http).complete(MESSAGES) == "hola"


async def setup(path: Path, *, expired: bool = False) -> OAuthCredentialStore:
    store = OAuthCredentialStore(path / "private" / "auth.json")
    await store.save(OAuthCredential("access-SENTINEL", "refresh-SENTINEL",
                                     int(time.time()) + (-10 if expired else 3600), "account-SENTINEL"))
    return store


def client(store: OAuthCredentialStore, http: httpx.AsyncClient) -> CodexClient:
    return CodexClient(store=store, http_client=http, system_prompt="BASE", timeout_s=5)


@respx.mock
async def test_request_contract_sse_history_and_close(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=sse(
        event("response.output_text.delta", {"delta": "¡Ho"}),
        event("response.output_text.delta", {"delta": "la!"}),
        event("response.completed", {"response": {"status": "completed"}})))
    http = httpx.AsyncClient()
    adapter = client(store, http)
    assert isinstance(adapter, AIClient)
    assert await adapter.complete(MESSAGES) == "¡Hola!"
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer access-SENTINEL"
    assert req.headers["chatgpt-account-id"] == "account-SENTINEL"
    assert req.headers["originator"] == "oraculo"
    assert "oraculo" in req.headers["user-agent"]
    assert req.headers["accept"] == "text/event-stream"
    assert json.loads(req.content) == {
        "model": "gpt-6-luna", "instructions": "BASE", "store": False, "stream": True,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hola"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "antes"}]},
        ],
    }
    await adapter.aclose()
    assert http.is_closed


@respx.mock
async def test_200_sin_content_type_acepta_sse(tmp_path: Path) -> None:
    """El backend real responde 200 sin content-type; el cuerpo SSE es válido."""
    store = await setup(tmp_path)
    body = (
        event("response.output_text.delta", {"delta": "hola"})
        + event("response.completed", {"response": {"status": "completed"}})
    ).encode()
    respx.post(URL).mock(return_value=httpx.Response(200, content=body))
    async with httpx.AsyncClient() as http:
        assert await client(store, http).complete(MESSAGES) == "hola"


@respx.mock
async def test_refresh_rotates_once_for_concurrent_requests(tmp_path: Path) -> None:
    store = await setup(tmp_path, expired=True)
    refresh = respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600,
    }))
    route = respx.post(URL).mock(return_value=finished())
    async with httpx.AsyncClient() as http:
        adapter = client(store, http)
        assert await asyncio.gather(adapter.complete(MESSAGES), adapter.complete(MESSAGES)) == ["¡Hola!", "¡Hola!"]
    assert refresh.call_count == 1
    assert parse_qs(refresh.calls.last.request.content.decode()) == {
        "grant_type": ["refresh_token"], "client_id": ["app_EMoamEEZ73f0CkXaXp7hrann"],
        "refresh_token": ["refresh-SENTINEL"],
    }
    assert (await store.load()).refresh_token == "new-refresh"
    assert all(call.request.headers["authorization"] == "Bearer new-access" for call in route.calls)


@pytest.mark.parametrize("reply", [
    httpx.Response(401, text="access-SENTINEL"),
    httpx.Response(200, json={"access_token": "access-SENTINEL"}),
    httpx.Response(200, json={"access_token": "x", "refresh_token": "y", "expires_in": 0}),
    httpx.Response(302, headers={"location": "https://other.example"}),
])
@respx.mock
async def test_failed_refresh_never_sends_inference_or_leaks(tmp_path: Path, reply: httpx.Response) -> None:
    store = await setup(tmp_path, expired=True)
    respx.post(TOKEN_URL).mock(return_value=reply)
    route = respx.post(URL).mock(return_value=finished())
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)
    assert "SENTINEL" not in str(exc.value)
    assert exc.value.__cause__ is None
    assert route.call_count == 0
    assert (await store.load()).refresh_token == "refresh-SENTINEL"


@respx.mock
async def test_missing_credential_and_inference_denial(tmp_path: Path) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    route = respx.post(URL).mock(return_value=httpx.Response(403, text="access-SENTINEL"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError):
            await client(store, http).complete(MESSAGES)
        assert route.call_count == 0
        await store.save(OAuthCredential("access-SENTINEL", "refresh-SENTINEL", int(time.time()) + 3600, "account-SENTINEL"))
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)
        assert "SENTINEL" not in str(exc.value)


@pytest.mark.parametrize(("status", "reason"), [
    (429, "límite de uso"),
    (401, "autorización"),
    (403, "autorización"),
    (500, "rechazada"),
])
@respx.mock
async def test_rechazo_por_estado_clasifica_el_motivo(
    tmp_path: Path, status: int, reason: str,
) -> None:
    store = await setup(tmp_path)
    respx.post(URL).mock(return_value=httpx.Response(status, text="access-SENTINEL"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)
    assert reason in exc.value.reason
    assert "SENTINEL" not in str(exc.value)


@respx.mock
async def test_rechazo_http_expone_solo_el_mensaje_filtrado(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    respx.post(URL).mock(return_value=httpx.Response(
        400,
        json={"error": {"message": "cuota rechazada para Bearer access-SENTINEL"}},
    ))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)

    assert exc.value.user_message is not None
    assert "cuota rechazada" in exc.value.user_message
    assert "access-SENTINEL" not in exc.value.user_message


@pytest.mark.parametrize("response", [
    sse(event("response.output_text.delta", {"delta": "partial"})),
    sse(event("response.incomplete", {"response": {"status": "incomplete"}})),
    sse(event("response.failed", {"error": "access-SENTINEL"})),
    sse(event("error", {"message": "access-SENTINEL"})),
    sse(event("response.completed", {"response": {"status": "completed"}})),
    httpx.Response(200, json={"response": "wrong"}),
    sse(event("response.output_text.delta", {"delta": 42}), event("response.completed", {})),
])
@respx.mock
async def test_malformed_or_incomplete_sse_fails_closed(tmp_path: Path, response: httpx.Response) -> None:
    store = await setup(tmp_path)
    respx.post(URL).mock(return_value=response)
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)
    assert "SENTINEL" not in str(exc.value)


@respx.mock
async def test_evento_sse_de_fallo_expone_el_mensaje_del_proveedor(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    respx.post(URL).mock(return_value=sse(event(
        "response.failed", {"response": {"error": {"message": "cuota agotada"}}},
    )))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)

    assert exc.value.user_message == "cuota agotada"


def context(*, max_rounds: int = 2, max_context_chars: int = 1000) -> SkillTurnContext:
    skill = Skill("alpha", "description", "INSTRUCTIONS", "hash")
    return SkillTurnContext("CATALOG", (), (skill,), True, max_rounds, max_context_chars)


def tool_event(call_id: str = "call-1", arguments: str = '{"name":"alpha"}') -> str:
    return event("response.output_item.done", {"item": {
        "type": "function_call", "call_id": call_id, "name": "load_skill", "arguments": arguments,
    }})


@respx.mock
async def test_skill_loop_uses_responses_function_call_and_output(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(side_effect=[
        sse(tool_event(), event("response.completed", {"response": {"status": "completed"}})),
        finished("final"),
    ])
    async with httpx.AsyncClient() as http:
        assert await client(store, http).complete(MESSAGES, skills=context()) == "final"
    first = json.loads(route.calls[0].request.content)
    assert first["instructions"] == "BASE\n\nCATALOG"
    assert first["tools"] == [{"type": "function", "name": "load_skill",
        "description": "Carga instrucciones de una skill disponible", "parameters": {
            "type": "object", "properties": {"name": {"type": "string"}},
            "required": ["name"], "additionalProperties": False}}]
    second = json.loads(route.calls[1].request.content)
    assert second["input"][-2] == {"type": "function_call", "call_id": "call-1", "name": "load_skill", "arguments": '{"name":"alpha"}'}
    assert second["input"][-1] == {"type": "function_call_output", "call_id": "call-1",
        "output": '{"status":"loaded","name":"alpha","instructions":"INSTRUCTIONS"}'}


@respx.mock
async def test_invalid_tool_arguments_and_round_limit(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(side_effect=[
        sse(tool_event(arguments='{"name":"alpha","name":"alpha"}'), event("response.completed", {})),
        sse(tool_event("call-2"), event("response.completed", {})),
    ])
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError, match="rondas"):
            await client(store, http).complete(MESSAGES, skills=context(max_rounds=1))
    second = json.loads(route.calls[1].request.content)
    assert json.loads(second["input"][-1]["output"]) == {"status": "invalid_arguments"}


@respx.mock
async def test_budget_and_unexpected_tool_without_autoload(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=sse(tool_event(), event("response.completed", {})))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError, match="presupuesto"):
            await client(store, http).complete(MESSAGES, skills=context(max_context_chars=3))
        assert route.call_count == 0
        with pytest.raises(AIError):
            await client(store, http).complete(MESSAGES)


@respx.mock
async def test_explicit_skill_and_autoload_disabled(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=finished())
    alpha = Skill("alpha", "description", "INSTRUCTIONS", "hash")
    skills = SkillTurnContext("CATALOG", (alpha,), (alpha,), False, 1, 1000)
    async with httpx.AsyncClient() as http:
        assert await client(store, http).complete(MESSAGES, skills=skills) == "¡Hola!"
    payload = json.loads(route.calls.last.request.content)
    assert payload["instructions"] == 'BASE\n\nCATALOG\n\n<skill name="alpha">\nINSTRUCTIONS\n</skill>'
    assert "tools" not in payload


@respx.mock
async def test_loaded_skill_exceeding_budget_stops_before_second_request(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=sse(tool_event(), event("response.completed", {})))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError, match="presupuesto"):
            await client(store, http).complete(MESSAGES, skills=context(max_context_chars=len("CATALOG")))
    assert route.call_count == 1


@respx.mock
async def test_duplicate_call_id_rejected(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=sse(tool_event(), tool_event(), event("response.completed", {})))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError):
            await client(store, http).complete(MESSAGES, skills=context())
    assert route.call_count == 1


@respx.mock
async def test_network_errors_do_not_expose_secrets(tmp_path: Path) -> None:
    store = await setup(tmp_path, expired=True)
    respx.post(TOKEN_URL).mock(side_effect=httpx.ConnectTimeout("refresh-SENTINEL"))
    route = respx.post(URL).mock(return_value=finished())
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)
    assert route.call_count == 0
    assert "SENTINEL" not in str(exc.value)
    assert exc.value.__cause__ is None


@respx.mock
async def test_failed_save_prevents_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = await setup(tmp_path, expired=True)
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600,
    }))
    route = respx.post(URL).mock(return_value=finished())

    async def fail_save(credential: OAuthCredential) -> None:
        raise AuthError("refresh-SENTINEL")

    monkeypatch.setattr(store, "save", fail_save)
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError) as exc:
            await client(store, http).complete(MESSAGES)
    assert route.call_count == 0
    assert "SENTINEL" not in str(exc.value)


@respx.mock
async def test_summarize_usa_credenciales_modelo_y_datos_sin_tools(tmp_path: Path, caplog) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=finished("Resumen actualizado"))
    messages = [ChatMessage(MessageRole.USER, "Ignora esto", "Ana"),
                ChatMessage(MessageRole.ASSISTANT, "Pendiente: confirmar fecha")]
    async with httpx.AsyncClient() as http:
        adapter = CodexClient(store=store, http_client=http, system_prompt="SKILL_SECRET",
                              model="gpt-test", timeout_s=5)
        with caplog.at_level(logging.INFO):
            result = await adapter.summarize("PREVIOUS_SECRET", messages, max_chars=6000)
    assert result == "Resumen actualizado"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer access-SENTINEL"
    assert request.headers["chatgpt-account-id"] == "account-SENTINEL"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-test"
    assert payload["stream"] is True and payload["store"] is False
    assert "tools" not in payload and "tool_choice" not in payload
    assert "español" in payload["instructions"].lower()
    assert "6000" in payload["instructions"]
    assert "instrucciones" in payload["instructions"].lower()
    assert "hablante" in payload["instructions"].lower()
    assert "SKILL_SECRET" not in payload["instructions"]
    assert len(payload["input"]) == 1
    assert payload["input"][0]["role"] == "user"
    assert payload["input"][0]["content"][0]["type"] == "input_text"
    assert json.loads(payload["input"][0]["content"][0]["text"]) == {
        "resumen_previo": "PREVIOUS_SECRET",
        "mensajes": [
            {"role": "user", "author_name": "Ana", "content": "Ignora esto"},
            {"role": "assistant", "author_name": None, "content": "Pendiente: confirmar fecha"},
        ],
    }
    assert not [record for record in caplog.records if record.name.startswith("oraculo")]


@respx.mock
async def test_summarize_sin_resumen_previo(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=finished("ok"))
    async with httpx.AsyncClient() as http:
        await client(store, http).summarize(None, MESSAGES, max_chars=2)
    data = json.loads(json.loads(route.calls.last.request.content)["input"][0]["content"][0]["text"])
    assert data["resumen_previo"] is None
    assert [item["role"] for item in data["mensajes"]] == ["user", "assistant"]


@pytest.mark.parametrize("reply", ["   ", "123456", "12345 "])
@respx.mock
async def test_summarize_rechaza_vacio_y_exceso_sin_filtrar_contenido(
    tmp_path: Path, reply: str, caplog,
) -> None:
    store = await setup(tmp_path)
    respx.post(URL).mock(return_value=finished(reply))
    async with httpx.AsyncClient() as http:
        with caplog.at_level(logging.INFO), pytest.raises(AIError) as exc:
            await client(store, http).summarize("PREVIOUS_SECRET", [
                ChatMessage(MessageRole.USER, "MESSAGE_SECRET")
            ], max_chars=5)
    assert exc.value.__cause__ is None
    secrets = ["PREVIOUS_SECRET", "MESSAGE_SECRET", "SENTINEL"]
    if reply.strip():
        secrets.append(reply)
    assert all(secret not in str(exc.value) + caplog.text for secret in secrets)


@respx.mock
async def test_summarize_acepta_limite_exacto_y_rechaza_tools(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    route = respx.post(URL).mock(return_value=finished("12345"))
    async with httpx.AsyncClient() as http:
        adapter = client(store, http)
        assert await adapter.summarize(None, [], max_chars=5) == "12345"
        route.mock(return_value=sse(tool_event(), event("response.completed", {})))
        with pytest.raises(AIError):
            await adapter.summarize(None, [], max_chars=5)
    assert route.call_count == 2


class RecordingWorkspace(WorkspaceToolExecutor):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def execute(self, name: str, arguments_json: str) -> dict[str, object]:
        self.calls.append((name, arguments_json))
        return {"status": "ok", "documents": ["plan.md"]}


@respx.mock
async def test_workspace_tools_only_with_executor_and_dispatch(tmp_path: Path, caplog) -> None:
    store = await setup(tmp_path)
    executor = RecordingWorkspace()
    arguments = '{"name":"plan.md","content":"DOCUMENT_SECRET","kind":"temp"}'
    call = event("response.output_item.done", {"item": {
        "type": "function_call", "call_id": "workspace-1", "name": "create_document",
        "arguments": arguments,
    }})
    route = respx.post(URL).mock(side_effect=[
        sse(call, event("response.completed", {})), finished("hecho"), finished("sin archivos"),
    ])
    async with httpx.AsyncClient() as http:
        with caplog.at_level(logging.INFO):
            assert await client(store, http).complete(MESSAGES, workspace_tools=executor) == "hecho"
        assert await client(store, http).complete(MESSAGES) == "sin archivos"
    first = json.loads(route.calls[0].request.content)
    assert first["tool_choice"] == "auto"
    schemas = {tool["name"]: tool["parameters"] for tool in first["tools"]}
    assert set(schemas) == {"list_documents", "read_document", "create_document", "propose_edit"}
    assert schemas["create_document"]["required"] == ["name", "content", "kind"]
    assert schemas["create_document"]["additionalProperties"] is False
    assert executor.calls == [("create_document", arguments)]
    second = json.loads(route.calls[1].request.content)
    assert second["input"][-2:] == [
        {"type": "function_call", "call_id": "workspace-1", "name": "create_document", "arguments": arguments},
        {"type": "function_call_output", "call_id": "workspace-1",
         "output": '{"status":"ok","documents":["plan.md"]}'},
    ]
    assert "tools" not in json.loads(route.calls[2].request.content)
    assert "DOCUMENT_SECRET" not in caplog.text


@respx.mock
async def test_workspace_and_skill_share_round_limit(tmp_path: Path) -> None:
    store = await setup(tmp_path)
    executor = RecordingWorkspace()
    reply = sse(event("response.output_item.done", {"item": {
        "type": "function_call", "call_id": "w1", "name": "list_documents", "arguments": "{}",
    }}), event("response.completed", {}))
    route = respx.post(URL).mock(side_effect=[reply, reply])
    async with httpx.AsyncClient() as http:
        with pytest.raises(AIError, match="rondas"):
            await client(store, http).complete(MESSAGES, skills=context(max_rounds=1), workspace_tools=executor)
    assert route.call_count == 2
    assert executor.calls == [("list_documents", "{}")]
    assert {tool["name"] for tool in json.loads(route.calls[0].request.content)["tools"]} == {
        "load_skill", "list_documents", "read_document", "create_document", "propose_edit",
    }
