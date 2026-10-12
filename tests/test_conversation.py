"""Tests del caso de uso `ConversationService`.

Cubre Req 1.1–1.6, 2.1–2.3, 3.1–3.5, 4.1–4.7, 6.3 y 7.x del spec
`sesiones-y-contexto`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from telegram.error import TelegramError

from fakes import (
    FakeAIClient,
    FakeAccessPolicy,
    FakeClock,
    FakePhraseProvider,
    FakeSessionRepository,
    group_message,
    private_message,
)

from oraculo.application.conversation import COMPACTION_NOTICE, ConversationService
from oraculo.application.sessions import SessionManager
from oraculo.application.skills import SkillService
from oraculo.application.workspace import WorkspaceService
from oraculo.infrastructure.workspace import FileDocumentStore
from oraculo.domain.errors import AIError
from oraculo.domain.models import (
    ChatKind,
    ChatMessage,
    IncomingMessage,
    Skill,
    SkillCatalogSnapshot,
)
from oraculo.domain.workspace import DocumentRef, WorkspaceScope
from oraculo.telegram.handlers import OraculoMessageHandler

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def build_service(
    ai_client: FakeAIClient | None = None,
    users: frozenset[int] = frozenset(),
    chats: frozenset[int] = frozenset(),
    phrase: str = "No hablo con extraños",
    max_turns: int = 15,
    idle_minutes: int = 240,
    skill_service: SkillService | None = None,
    workspace_service: WorkspaceService | None = None,
) -> tuple[
    ConversationService, FakeAIClient, FakePhraseProvider, FakeSessionRepository, FakeClock
]:
    ai = ai_client if ai_client is not None else FakeAIClient()
    phrases = FakePhraseProvider(rejection=phrase)
    repository = FakeSessionRepository()
    clock = FakeClock(NOW)
    sessions = SessionManager(
        repository=repository,
        max_turns=max_turns,
        idle_minutes=idle_minutes,
        clock=clock,
    )
    service = ConversationService(
        ai_client=ai,
        access_policy=FakeAccessPolicy(users, chats),
        phrase_provider=phrases,
        sessions=sessions,
        skill_service=skill_service
        if skill_service is not None
        else make_skill_service(),
        workspace_service=workspace_service,
    )
    return service, ai, phrases, repository, clock


async def test_workspace_turn_and_local_confirmation_do_not_persist_documents(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset({-100}), frozenset({-100}))

    class ToolAI(FakeAIClient):
        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            assert workspace_tools is not None
            await workspace_tools.execute("create_document", '{"name":"secret.md","content":"SECRETO","kind":"final"}')
            proposal = await workspace_tools.execute("propose_edit", '{"name":"secret.md","content":"CAMBIO"}')
            self.code = proposal["code"]
            return "Listo"

    ai = ToolAI()
    service, _, _, repo, _ = build_service(ai_client=ai, users=frozenset({1}), workspace_service=workspace)
    reply = await service.handle(private_message("trabaja"))
    assert "secret.md" in reply and "CAMBIO" in reply
    assert "SECRETO" not in str(repo.sessions)
    assert "CAMBIO" not in str(repo.sessions) and ai.code not in str(repo.sessions)
    service.preview_delivery(private_message("trabaja"), reply, True)
    assert "actualizado" in (await service.handle(private_message(f"confirma {ai.code}")) or "")
    assert len(repo.appended) == 1
    await service.handle_reset(private_message())


async def test_workspace_attachment_respects_conversation_gate(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset({-100}), frozenset({-100}))
    from oraculo.domain.workspace import DocumentRef, WorkspaceScope

    await workspace._store.create(WorkspaceScope(ChatKind.GROUP, -100), DocumentRef("a.md"), "grupo")
    service, _, _, _, _ = build_service(chats=frozenset({-100}), workspace_service=workspace)
    assert await service.handle_attachment(group_message(text="adjunta a.md", mentions_bot=False)) is None
    assert await service.handle_attachment(group_message(text="adjunta a.md", user_id=2)) == ("a.md", b"grupo")
    assert await service.handle_attachment(group_message(text="adjunta a.md")) == ("a.md", b"grupo")


async def test_ambiguous_attachment_is_local_and_does_not_read_or_call_ai(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset({-100}), frozenset({-100}))
    async def forbidden(*args):
        raise AssertionError("No document IO for ambiguous attachment")
    monkeypatch.setattr(workspace._store, "read", forbidden)
    monkeypatch.setattr(workspace._store, "attachment", forbidden)
    service, ai, _, repo, _ = build_service(users=frozenset({1}), chats=frozenset({-100}), workspace_service=workspace)
    for request in (private_message("envíame"), private_message("envíame a.md y b.md"),
                    group_message("adjunta a.md.pdf")):
        assert await service.handle_attachment(request) is None
        reply = await service.handle(request)
        assert ".md" in reply and ".txt" in reply
    denied = await service.handle(group_message("envíame secreto.md", user_id=2))
    assert "secreto.md" not in (denied or "")
    assert ai.calls == [] and repo.appended == []


async def test_nonexistent_named_attachment_does_not_fall_back_to_ai(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset(), frozenset())
    service, ai, _, repo, _ = build_service(users=frozenset({1}), workspace_service=workspace)
    request = private_message("envíame ausente.md")
    assert await service.handle_attachment(request) is None
    assert ".md" in (await service.handle(request) or "")
    assert ai.calls == [] and repo.appended == []


async def test_handler_ambiguous_attachment_with_real_service_never_sends_document(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset(), frozenset())
    service, ai, _, _, _ = build_service(users=frozenset({1}), workspace_service=workspace)

    class Message:
        text = "envíame a.md y b.md"
        entities = ()
        caption_entities = ()
        reply_to_message = None
        message_thread_id = None

        def __init__(self):
            self.replies = []
            self.documents = []

        async def reply_text(self, text):
            self.replies.append(text)

        async def reply_document(self, document, *, filename):
            self.documents.append(filename)

    message = Message()
    update = SimpleNamespace(effective_message=message,
                             effective_chat=SimpleNamespace(id=100, type="private"),
                             effective_user=SimpleNamespace(id=1, is_bot=False, first_name="Ana"))
    context = SimpleNamespace(bot=SimpleNamespace(id=42, username="oraculo_bot"))
    await OraculoMessageHandler(service)(update, context)
    assert message.documents == []
    assert len(message.replies) == 1 and ".md" in message.replies[0]
    assert ai.calls == []


async def test_long_document_or_long_ai_reply_offers_private_attachment(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset({-100}), frozenset({-100}))
    await workspace._store.create(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("largo.md"), "X" * 1901)
    await workspace._store.create(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("corto.txt"), "breve")

    class ReadingAI(FakeAIClient):
        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            name = "largo.md" if "largo.md" in messages[-1].content else "corto.txt"
            result = await workspace_tools.execute("read_document", '{"name":"' + name + '"}')
            if name == "largo.md":
                assert result["content"] == "X" * 1901
                return "Respuesta breve"
            return "Y" * 1901

    service, _, _, _, _ = build_service(ai_client=ReadingAI(), users=frozenset({1}), workspace_service=workspace)
    for name in ("largo.md", "corto.txt"):
        reply = await service.handle(private_message("lee " + name))
        assert "envíame " + name in reply
        if name == "corto.txt":
            assert "Y" * 1901 in reply
        assert await service.handle_attachment(private_message("envíame " + name)) is not None


async def test_long_group_read_asks_to_narrow_without_automatic_attachment(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset(), frozenset({-100}), frozenset({-100}))
    await workspace._store.create(WorkspaceScope(ChatKind.GROUP, -100), DocumentRef("largo.md"), "X" * 1901)

    class ReadingAI(FakeAIClient):
        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            await workspace_tools.execute("read_document", '{"name":"largo.md"}')
            return "Y" * 1901

    service, _, _, _, _ = build_service(ai_client=ReadingAI(), chats=frozenset({-100}), workspace_service=workspace)
    reply = await service.handle(group_message("lee largo.md"))
    assert "Y" * 1901 not in reply
    assert "acota la consulta" in reply
    assert "envíame" not in reply


async def test_group_workspace_tools_require_current_member_explicit_request(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset(), frozenset({-100}), frozenset({-100}))

    class SpyAI(FakeAIClient):
        def __init__(self):
            super().__init__()
            self.tools = []

        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            self.tools.append(workspace_tools)
            return await super().complete(messages, skills=skills)

    ai = SpyAI()
    service, _, _, _, _ = build_service(ai_client=ai, chats=frozenset({-100}), workspace_service=workspace)
    await service.handle(group_message("guarda un documento con secretos", user_id=2))
    await service.handle(group_message("hola", user_id=1))
    await service.handle(group_message("hola", user_id=1, mentions_bot=False,
                                       replies_to_bot=True, quoted_text="Voy a editar el documento secreto.md"))
    await service.handle(group_message("lee el documento secreto.md", user_id=2))
    await service.handle(group_message("lee el documento secreto.md", user_id=1))
    assert ai.tools[1] is None
    assert ai.tools[2] is None
    assert ai.tools[0] is not None
    assert ai.tools[3] is not None
    assert ai.tools[4] is not None
    assert contents(ai.calls[3]) == ["lee el documento secreto.md"]
    assert contents(ai.calls[4]) == ["lee el documento secreto.md"]


async def test_group_workspace_turn_excludes_quoted_bot_text(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset(), frozenset({-100}), frozenset({-100}))

    class SpyAI(FakeAIClient):
        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            assert workspace_tools is not None
            assert len(messages) == 1
            assert "instrucción antigua" not in messages[0].content
            return "ok"

    service, _, _, _, _ = build_service(ai_client=SpyAI(), chats=frozenset({-100}), workspace_service=workspace)
    assert await service.handle(group_message("lee documento plan.md", mentions_bot=False,
                                              replies_to_bot=True, quoted_text="instrucción antigua")) == "ok"


async def test_partial_telegram_preview_failure_cannot_be_confirmed(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset(), frozenset())
    scope = WorkspaceScope(ChatKind.PRIVATE, 1)
    await workspace._store.create(scope, DocumentRef("plan.md"), "a" * 3000)

    class ProposingAI(FakeAIClient):
        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            result = await workspace_tools.execute(
                "propose_edit", '{"name":"plan.md","content":"' + "b" * 3000 + '"}'
            )
            self.code = result["code"]
            return "Vista previa"

    class PartialMessage:
        text = "edita plan.md"
        entities = ()
        caption_entities = ()
        reply_to_message = None
        message_thread_id = None

        def __init__(self):
            self.sent = []

        async def reply_text(self, text):
            if self.sent:
                raise TelegramError("entrega fallida")
            self.sent.append(text)

    ai = ProposingAI()
    service, _, _, _, _ = build_service(ai_client=ai, users=frozenset({1}), workspace_service=workspace)
    message = PartialMessage()
    update = SimpleNamespace(effective_message=message,
                             effective_chat=SimpleNamespace(id=1, type="private"),
                             effective_user=SimpleNamespace(id=1, is_bot=False, first_name="Ana"))
    context = SimpleNamespace(bot=SimpleNamespace(id=42, username="oraculo_bot"))
    await OraculoMessageHandler(service)(update, context)
    assert len(message.sent) == 1
    assert "no se aplicó" in (await service.handle(private_message(f"confirma {ai.code}")) or "")
    assert (await workspace._store.read(scope, DocumentRef("plan.md")))[0] == "a" * 3000


async def test_workspace_ai_failure_reports_committed_creation(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    workspace = WorkspaceService(FileDocumentStore(root), frozenset({1}), frozenset(), frozenset())

    class FailingAI(FakeAIClient):
        async def complete(self, messages, *, skills, workspace_tools=None, attachments=()):
            await workspace_tools.execute("create_document", '{"name":"saved.md","content":"dato","kind":"final"}')
            raise AIError("unavailable")

    service, _, _, repo, _ = build_service(ai_client=FailingAI(), users=frozenset({1}), workspace_service=workspace)
    reply = await service.handle(private_message("guarda"))
    assert "saved.md" in reply
    assert repo.appended == []
    assert await service.handle_attachment(private_message("envíame saved.md")) == ("saved.md", b"dato")


def make_skill(name: str, instructions: str = "INSTRUCCIONES") -> Skill:
    return Skill(name, f"Descripción {name}", instructions, f"hash-{name}")


def make_skill_service(
    *skills: Skill,
    autoload: bool = True,
    max_per_turn: int = 3,
    max_context_chars: int = 40_000,
) -> SkillService:
    return SkillService(
        SkillCatalogSnapshot(skills),
        autoload_enabled=autoload,
        catalog_max_chars=8_000,
        max_context_chars=max_context_chars,
        max_per_turn=max_per_turn,
        max_rounds=3,
    )


def contents(messages: tuple[ChatMessage, ...]) -> list[str]:
    return [message.content for message in messages]


# --- Privado: contexto y ventana (Req 1.1–1.6) ---


async def test_privado_permitido_consulta_la_ia():
    service, ai, _, _, _ = build_service(users=frozenset({1}))

    reply = await service.handle(private_message(text="hola"))

    assert reply == "respuesta IA"
    assert contents(ai.calls[0]) == ["hola"]


async def test_privado_guarda_el_turno_en_su_sesion():
    service, _, _, repository, _ = build_service(users=frozenset({1}))

    await service.handle(private_message(text="hola"))

    assert len(repository.appended) == 1
    key, user_message, assistant_message = repository.appended[0]
    assert key == "private:1"
    assert user_message.content == "hola"
    assert assistant_message.content == "respuesta IA"


async def test_el_segundo_mensaje_incluye_el_historial():
    service, ai, _, _, _ = build_service(users=frozenset({1}))

    await service.handle(private_message(text="hola"))
    await service.handle(private_message(text="otra vez"))

    assert contents(ai.calls[1]) == ["hola", "respuesta IA", "otra vez"]


async def test_la_ventana_limita_lo_reenviado():
    service, ai, _, _, _ = build_service(users=frozenset({1}), max_turns=1)

    await service.handle(private_message(text="m1"))
    await service.handle(private_message(text="m2"))
    await service.handle(private_message(text="m3"))

    assert "resumen generado" in ai.calls[2][0].content
    assert contents(ai.calls[2][1:]) == ["m3"]


async def test_compacta_despues_del_umbral_y_omite_aviso_del_contexto():
    service, ai, _, repo, _ = build_service(users=frozenset({1}), max_turns=2)
    assert await service.handle(private_message("uno")) == "respuesta IA"
    reply = await service.handle(private_message("dos"))
    assert "respuesta IA" in reply and "resum" in reply.lower()
    assert len(ai.summaries) == 1
    previous, sent, maximum = ai.summaries[0]
    assert previous is None and maximum == 6000
    assert contents(sent) == ["uno", "respuesta IA", "dos", "respuesta IA"]
    assert repo.sessions["private:1"].messages == ()
    assert repo.sessions["private:1"].summary == "resumen generado"
    await service.handle(private_message("tres"))
    assert "resumen generado" in ai.calls[2][0].content
    assert contents(ai.calls[2][1:]) == ["tres"]
    assert all("compact" not in m.content.lower() for call in ai.calls for m in call)


async def test_segundo_ciclo_incluye_resumen_previo_y_reinicio_borra_ambos():
    service, ai, _, repo, _ = build_service(users=frozenset({1}), max_turns=2)
    for word in ("uno", "dos", "tres", "cuatro"):
        await service.handle(private_message(word))
    assert [entry[0] for entry in ai.summaries] == [None, "resumen generado"]
    assert contents(ai.summaries[1][1]) == ["tres", "respuesta IA", "cuatro", "respuesta IA"]
    await service.handle_reset(private_message())
    assert "private:1" not in repo.sessions
    await service.handle(private_message("nuevo"))
    assert contents(ai.calls[-1]) == ["nuevo"]


async def test_error_al_resumir_conserva_turnos_y_reintenta_antes_del_siguiente():
    ai = FakeAIClient()
    ai.summary_error = AIError("red fallida")
    service, _, _, repo, _ = build_service(ai_client=ai, users=frozenset({1}), max_turns=2)
    await service.handle(private_message("uno"))
    reply = await service.handle(private_message("dos"))
    assert reply == "respuesta IA"
    assert repo.sessions["private:1"].summary is None
    assert len(repo.sessions["private:1"].messages) == 4
    ai.summary_error = None
    reply = await service.handle(private_message("tres"))
    assert len(ai.summaries) == 2
    assert "resum" in reply.lower()
    assert "resumen generado" in ai.calls[-1][0].content
    assert contents(ai.calls[-1][1:]) == ["tres"]
    assert len(repo.sessions["private:1"].messages) == 2


async def test_grupo_compacta_una_vez_bajo_concurrencia():
    service, ai, _, repo, _ = build_service(chats=frozenset({-100}), max_turns=2)
    replies = await asyncio.gather(
        service.handle(group_message("uno", user_display_name="Ana")),
        service.handle(group_message("dos", user_display_name="Luis")),
    )
    assert len(ai.summaries) == 1
    assert contents(ai.summaries[0][1]) == ["Ana: uno", "respuesta IA", "Luis: dos", "respuesta IA"]
    assert sum("resum" in reply.lower() for reply in replies) == 1
    assert repo.sessions["group:-100"].messages == ()


async def test_resumen_invalido_no_borra_historial_ni_anuncia_exito(caplog):
    ai = FakeAIClient()
    ai.summary_reply = "X" * 6001
    service, _, _, repo, _ = build_service(ai_client=ai, users=frozenset({1}), max_turns=1)
    with caplog.at_level(logging.WARNING):
        reply = await service.handle(private_message("dato privado"))
    assert reply == "respuesta IA"
    assert len(repo.sessions["private:1"].messages) == 2
    assert repo.sessions["private:1"].summary is None
    assert "dato privado" not in caplog.text
    assert "XXXX" not in caplog.text


async def test_reintento_fallido_no_duplica_llamadas_en_el_mismo_turno():
    ai = FakeAIClient()
    ai.summary_error = AIError("sin red")
    service, _, _, repo, _ = build_service(ai_client=ai, users=frozenset({1}), max_turns=1)
    await service.handle(private_message("uno"))
    reply = await service.handle(private_message("dos"))
    assert reply == "respuesta IA"
    assert len(ai.summaries) == 2
    assert len(repo.sessions["private:1"].messages) == 4
    assert repo.sessions["private:1"].summary is None


async def test_memoria_desactivada_no_resumira():
    service, ai, _, repo, _ = build_service(users=frozenset({1}), max_turns=0)
    await service.handle(private_message("uno"))
    await service.handle(private_message("dos"))
    assert ai.summaries == []
    assert repo.sessions == {}


async def test_error_de_ia_no_guarda_el_turno():
    ai = FakeAIClient(error=AIError("tiempo de espera agotado"))
    service, _, phrases, repository, _ = build_service(
        ai_client=ai, users=frozenset({1})
    )

    reply = await service.handle(private_message())

    assert reply == "OOps: tiempo de espera agotado"
    assert not hasattr(phrases, "ai_error_phrase")
    assert repository.appended == []


async def test_error_de_proveedor_usa_el_mensaje_publico_en_lugar_del_motivo_interno():
    ai = FakeAIClient(
        error=AIError("estado HTTP 429", user_message="cuota agotada")
    )
    service, _, _, repository, _ = build_service(
        ai_client=ai, users=frozenset({1})
    )

    reply = await service.handle(private_message())

    assert reply == "OOps: cuota agotada"
    assert "estado HTTP 429" not in reply
    assert repository.appended == []


async def test_error_sin_detalle_usa_fallback_sin_inventar_una_causa():
    ai = FakeAIClient(error=AIError("", user_message=""))
    service, _, _, repository, _ = build_service(
        ai_client=ai, users=frozenset({1})
    )

    reply = await service.handle(private_message())

    assert reply == "OOps: No hay más detalles disponibles."
    assert repository.appended == []


async def test_error_conserva_avisos_de_herramientas():
    async def no_confirmation(_message):
        return None

    def no_attachment_clarification(_message):
        return None

    workspace = SimpleNamespace(
        confirm=no_confirmation,
        attachment_clarification=no_attachment_clarification,
        workspace_for=lambda _message: SimpleNamespace(notices=["aviso interno"]),
    )
    ai = FakeAIClient(error=AIError("tiempo de espera agotado"))
    service, _, _, repository, _ = build_service(
        ai_client=ai, users=frozenset({1}), workspace_service=workspace,  # type: ignore[arg-type]
    )

    reply = await service.handle(private_message("lee el documento"))

    assert reply == "OOps: tiempo de espera agotado\n\naviso interno"
    assert repository.appended == []


async def test_error_conserva_aviso_de_compactacion():
    ai = FakeAIClient()
    ai.summary_error = AIError("fallo de resumen")
    service, _, _, repository, _ = build_service(
        ai_client=ai, users=frozenset({1}), max_turns=1,
    )
    assert await service.handle(private_message("primero")) == "respuesta IA"
    ai.summary_error = None
    ai.error = AIError("tiempo de espera agotado")

    reply = await service.handle(private_message("segundo"))

    assert reply == f"OOps: tiempo de espera agotado\n\n{COMPACTION_NOTICE}"
    assert len(repository.appended) == 1


async def test_skill_explicita_se_envia_sin_persistir_su_cuerpo():
    alpha = make_skill("alpha", "CUERPO PRIVADO")
    service, ai, _, repository, _ = build_service(
        users=frozenset({1}), skill_service=make_skill_service(alpha)
    )

    await service.handle(private_message(text="$alpha resume esto"))

    assert ai.skill_contexts[0].explicit_skills == (alpha,)
    _, stored_user, stored_assistant = repository.appended[0]
    assert stored_user.content == "$alpha resume esto"
    assert "CUERPO PRIVADO" not in stored_user.content
    assert stored_assistant.content == "respuesta IA"


async def test_autoload_se_ofrece_solo_en_el_contexto_efimero():
    alpha = make_skill("alpha")
    service, ai, _, repository, _ = build_service(
        users=frozenset({1}), skill_service=make_skill_service(alpha)
    )

    await service.handle(private_message(text="consulta ordinaria"))

    assert ai.skill_contexts[0].loadable_skills == (alpha,)
    assert ai.skill_contexts[0].autoload_enabled is True
    assert repository.appended[0][1].content == "consulta ordinaria"


async def test_limite_de_skills_responde_localmente_sin_ia_ni_persistencia():
    service, ai, phrases, repository, _ = build_service(
        users=frozenset({1}),
        skill_service=make_skill_service(
            make_skill("alpha"), make_skill("beta"), max_per_turn=1
        ),
    )

    reply = await service.handle(private_message(text="$alpha $beta consulta"))

    assert reply == phrases.skill_limit
    assert phrases.skill_limit_calls == 1
    assert ai.calls == []
    assert repository.appended == []


async def test_acceso_rechazado_se_evalua_antes_del_limite_de_skills():
    service, ai, phrases, repository, _ = build_service(
        users=frozenset({1}),
        skill_service=make_skill_service(
            make_skill("alpha"), make_skill("beta"), max_per_turn=1
        ),
    )

    reply = await service.handle(private_message(text="$alpha $beta", user_id=99))

    assert reply == phrases.rejection
    assert phrases.skill_limit_calls == 0
    assert ai.calls == []
    assert repository.appended == []


async def test_snapshot_vacio_conserva_contexto_neutral():
    service, ai, _, _, _ = build_service(users=frozenset({1}))

    await service.handle(private_message(text="hola"))

    context = ai.skill_contexts[0]
    assert context.catalog_text == ""
    assert context.explicit_skills == ()
    assert context.loadable_skills == ()
    assert context.autoload_enabled is False


async def test_mensajes_concurrentes_del_mismo_chat_se_serializan():
    service, ai, _, repository, _ = build_service(users=frozenset({1}))

    await asyncio.gather(
        service.handle(private_message(text="uno")),
        service.handle(private_message(text="dos")),
    )

    assert len(ai.calls) == 2
    assert contents(ai.calls[1]) == ["uno", "respuesta IA", "dos"]
    assert len(repository.appended) == 2


async def test_privado_no_permitido_rechaza_sin_llamar_a_la_ia():
    service, ai, phrases, repository, _ = build_service(users=frozenset({1}))

    reply = await service.handle(private_message(user_id=99))

    assert reply == "No hablo con extraños"
    assert phrases.rejection_calls == 1
    assert ai.calls == []
    assert repository.appended == []


async def test_texto_vacio_se_ignora():
    service, ai, _, repository, _ = build_service(users=frozenset({1}))

    reply = await service.handle(private_message(text="   "))

    assert reply is None
    assert ai.calls == []
    assert repository.appended == []


async def test_mensaje_no_texto_avisa_sin_llamar_a_la_ia():
    service, ai, phrases, _, _ = build_service(users=frozenset({1}))

    reply = await service.handle(private_message(text=None))

    assert reply == phrases.only_text
    assert phrases.only_text_calls == 1
    assert ai.calls == []


async def test_otro_tipo_de_chat_se_ignora():
    service, ai, _, _, _ = build_service(users=frozenset({1}))
    message = IncomingMessage(
        chat_id=1,
        user_id=1,
        chat_type=ChatKind.OTHER,
        text="hola",
        mentions_bot=False,
        replies_to_bot=False,
        quoted_text=None,
    )

    reply = await service.handle(message)

    assert reply is None
    assert ai.calls == []


# --- Grupos: sesión compartida y atribución (Req 2.1–2.4) ---


async def test_grupo_sin_mencion_se_ignora():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(group_message(mentions_bot=False))

    assert reply is None
    assert ai.calls == []


async def test_grupo_no_permitido_se_ignora():
    service, ai, _, _, _ = build_service(chats=frozenset({-999}))

    reply = await service.handle(group_message(chat_id=-100))

    assert reply is None
    assert ai.calls == []


async def test_grupo_con_mencion_consulta_la_ia():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(group_message(text="dime algo"))

    assert reply == "respuesta IA"
    assert contents(ai.calls[0]) == ["dime algo"]


async def test_grupo_usa_una_sesion_compartida_entre_miembros():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    await service.handle(
        group_message(text="¿qué es Kafka?", user_id=1, user_display_name="Ana")
    )
    await service.handle(
        group_message(text="¿y para qué sirve?", user_id=2, user_display_name="Luis")
    )

    assert contents(ai.calls[1]) == [
        "Ana: ¿qué es Kafka?",
        "respuesta IA",
        "Luis: ¿y para qué sirve?",
    ]


async def test_grupo_atribuye_los_mensajes_con_el_nombre():
    service, ai, _, repository, _ = build_service(chats=frozenset({-100}))

    await service.handle(group_message(text="dime algo", user_display_name="Ana"))

    assert contents(ai.calls[0]) == ["Ana: dime algo"]
    _, stored_user, _ = repository.appended[0]
    assert stored_user.content == "dime algo"
    assert stored_user.author_name == "Ana"


async def test_privado_no_atribuye_con_nombre():
    service, ai, _, _, _ = build_service(users=frozenset({1}))

    await service.handle(private_message(text="hola", user_display_name="Ana"))

    assert contents(ai.calls[0]) == ["hola"]


async def test_grupo_solo_mencion_responde_sin_llamar_a_la_ia():
    service, ai, phrases, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(group_message(text="", mentions_bot=True))

    assert reply == phrases.empty_mention
    assert phrases.empty_mention_calls == 1
    assert ai.calls == []


async def test_grupo_texto_vacio_sin_mencion_se_ignora():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(group_message(text="   ", mentions_bot=False))

    assert reply is None
    assert ai.calls == []


async def test_grupo_mensaje_no_texto_avisa_sin_llamar_a_la_ia():
    service, ai, phrases, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(group_message(text=None, mentions_bot=True))

    assert reply == phrases.only_text
    assert phrases.only_text_calls == 1
    assert ai.calls == []


async def test_grupo_error_de_ia_devuelve_prefijo_y_motivo():
    ai = FakeAIClient(error=AIError("tiempo de espera agotado"))
    service, _, phrases, _, _ = build_service(ai_client=ai, chats=frozenset({-100}))

    reply = await service.handle(group_message(text="dime algo"))

    assert reply == "OOps: tiempo de espera agotado"
    assert not hasattr(phrases, "ai_error_phrase")


# --- Contexto citado como refuerzo (Req 7.3) ---


async def test_grupo_respuesta_al_bot_consulta_la_ia_sin_mencion():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(
        group_message(
            text="¿y un ejemplo?",
            mentions_bot=False,
            replies_to_bot=True,
            quoted_text="La entropía es una medida de desorden.",
        )
    )

    assert reply == "respuesta IA"
    assert len(ai.calls) == 1


async def test_prompt_incluye_el_contexto_citado():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    await service.handle(
        group_message(
            text="¿y un ejemplo?",
            mentions_bot=False,
            replies_to_bot=True,
            quoted_text="La entropía es una medida de desorden.",
        )
    )

    prompt = ai.calls[0][0].content
    assert "La entropía es una medida de desorden." in prompt
    assert "¿y un ejemplo?" in prompt


async def test_el_contexto_citado_no_se_guarda_en_el_historial():
    service, _, _, repository, _ = build_service(chats=frozenset({-100}))

    await service.handle(
        group_message(
            text="¿y un ejemplo?",
            mentions_bot=False,
            replies_to_bot=True,
            quoted_text="La entropía es una medida de desorden.",
        )
    )

    _, stored_user, _ = repository.appended[0]
    assert stored_user.content == "¿y un ejemplo?"
    assert "Contexto:" not in stored_user.content


async def test_prompt_sin_contexto_cuando_no_hay_texto_citado():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    await service.handle(
        group_message(
            text="¿y un ejemplo?",
            mentions_bot=True,
            replies_to_bot=True,
            quoted_text=None,
        )
    )

    assert contents(ai.calls[0]) == ["¿y un ejemplo?"]


async def test_grupo_respuesta_a_otro_usuario_se_ignora():
    service, ai, _, _, _ = build_service(chats=frozenset({-100}))

    reply = await service.handle(
        group_message(
            text="de acuerdo",
            mentions_bot=False,
            replies_to_bot=False,
            quoted_text=None,
        )
    )

    assert reply is None
    assert ai.calls == []


# --- Expiración y apagador (Req 3.1, 3.3, 6.3) ---


async def test_la_sesion_expirada_arranca_limpia():
    service, ai, _, repository, clock = build_service(
        users=frozenset({1}), idle_minutes=240
    )
    await service.handle(private_message(text="hola"))

    clock.advance(timedelta(minutes=241))
    await service.handle(private_message(text="otra vez"))

    assert contents(ai.calls[1]) == ["otra vez"]
    assert repository.discarded == ["private:1"]


async def test_n_cero_no_guarda_ni_reenvia_historial():
    service, ai, _, repository, _ = build_service(users=frozenset({1}), max_turns=0)

    await service.handle(private_message(text="hola"))
    await service.handle(private_message(text="otra vez"))

    assert contents(ai.calls[1]) == ["otra vez"]
    assert repository.appended == []


# --- Reinicio de sesión con /new (Req 4.1–4.7) ---


async def test_reset_privado_descarta_y_confirma():
    service, ai, phrases, repository, _ = build_service(users=frozenset({1}))
    await service.handle(private_message(text="hola"))

    reply = await service.handle_reset(private_message())

    assert reply == phrases.new_session
    assert phrases.new_session_calls == 1
    assert repository.discarded == ["private:1"]
    assert len(ai.calls) == 1  # el reinicio no consulta la IA


async def test_reset_deja_la_sesion_vacia():
    service, ai, _, _, _ = build_service(users=frozenset({1}))
    await service.handle(private_message(text="hola"))

    await service.handle_reset(private_message())
    await service.handle(private_message(text="otra vez"))

    assert contents(ai.calls[1]) == ["otra vez"]


async def test_reset_privado_no_permitido_rechaza_sin_tocar_sesiones():
    service, ai, phrases, repository, _ = build_service(users=frozenset({1}))

    reply = await service.handle_reset(private_message(user_id=99))

    assert reply == "No hablo con extraños"
    assert phrases.rejection_calls == 1
    assert repository.discarded == []
    assert ai.calls == []


async def test_reset_grupo_descarta_la_sesion_compartida():
    service, _, phrases, repository, _ = build_service(chats=frozenset({-100}))
    await service.handle(group_message(text="dime algo"))

    reply = await service.handle_reset(group_message())

    assert reply == phrases.new_session
    assert repository.discarded == ["group:-100"]


async def test_reset_grupo_no_permitido_se_ignora():
    service, ai, _, repository, _ = build_service(chats=frozenset({-999}))

    reply = await service.handle_reset(group_message(chat_id=-100))

    assert reply is None
    assert repository.discarded == []
    assert ai.calls == []


async def test_reset_sin_sesion_previa_confirma_igual():
    service, ai, phrases, _, _ = build_service(users=frozenset({1}))

    reply = await service.handle_reset(private_message())

    assert reply == phrases.new_session
    assert ai.calls == []


# --- Bugfix modo-pi/solo-chat-privado: acceso al modo Pi (Req 3, 4, 6) ---

from oraculo.application.conversation import PI_GROUP_NOTICE
from oraculo.domain.pi import PiAccess


def test_pi_access_allows_authorized_private_user():
    service, *_ = build_service(users=frozenset({1}))
    assert service.pi_access(private_message("/agent", user_id=1)) == PiAccess(True, None)


def test_pi_access_rejects_unauthorized_private_user():
    service, *_ = build_service(users=frozenset({1}), phrase="Fuera")
    assert service.pi_access(private_message("/agent", user_id=2)) == PiAccess(False, "Fuera")


def test_pi_access_notifies_in_authorized_group():
    service, *_ = build_service(users=frozenset({1}), chats=frozenset({-100}))
    access = service.pi_access(group_message("/agent", chat_id=-100, mentions_bot=False))
    assert access == PiAccess(False, PI_GROUP_NOTICE)


def test_pi_access_ignores_unauthorized_group():
    service, *_ = build_service(users=frozenset({1}))
    assert service.pi_access(group_message("/agent", chat_id=-200)) == PiAccess(False, None)


def test_pi_access_ignores_other_chat_kinds():
    service, *_ = build_service(users=frozenset({1}))
    message = IncomingMessage(
        chat_id=-300, user_id=1, chat_type=ChatKind.OTHER, text="/agent",
        mentions_bot=False, replies_to_bot=False, quoted_text=None,
        user_display_name=None,
    )
    assert service.pi_access(message) == PiAccess(False, None)
