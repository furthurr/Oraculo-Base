import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fakes import group_message, private_message
from oraculo.application.workspace import WorkspaceService
from oraculo.domain.models import ChatKind, IncomingMessage
from oraculo.domain.workspace import DocumentRef, WorkspaceError, WorkspaceScope
from oraculo.infrastructure.workspace import FileDocumentStore


@pytest.fixture
def service(tmp_path: Path) -> WorkspaceService:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    return WorkspaceService(FileDocumentStore(root), frozenset({1, 2}), frozenset({-100}), frozenset({-100}))


def message(text: str, *, user: int = 1, chat: int = 1) -> IncomingMessage:
    return IncomingMessage(chat, user, ChatKind.PRIVATE, text, False, False, None)


async def call(turn, tool: str, **kwargs):
    return await turn.execute(tool, json.dumps(kwargs))


async def delivered_proposal(service, actor, name, content):
    turn = service.workspace_for(actor)
    result = await call(turn, "propose_edit", name=name, content=content)
    service.preview_delivery(actor, "\n".join(turn.notices), True)
    return result["code"]


async def test_scope_and_forged_calls_are_denied(service):
    assert service.workspace_for(message("x", user=3)) is None
    assert service._scope(group_message(user_id=2)) is not None
    assert service.workspace_for(group_message(text="lee documento a.md", user_id=2)) is not None
    assert service.workspace_for(group_message(text="lee documento a.md", user_id=1)) is not None
    turn = service.workspace_for(message("x"))
    assert (await turn.execute("create_document", '{"name":"a.md","name":"b.md","content":"x"}'))["error"] == "invalid_arguments"
    assert (await call(turn, "delete_document", name="a.md"))["error"] == "unknown_tool"
    assert (await call(turn, "create_document", name="../escape.md", content="x"))["error"] == "invalid_reference"
    assert (await call(turn, "create_document", name="a.md", content="x", scope="group"))["error"] == "invalid_arguments"
    assert turn.notices


async def test_any_group_member_can_use_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    service = WorkspaceService(FileDocumentStore(root), frozenset(), frozenset({-100}), frozenset({-100}))
    creator = group_message(text="crea el documento equipo.md con el plan", user_id=2)
    turn = service.workspace_for(creator)
    assert turn is not None
    assert (await call(turn, "create_document", name="equipo.md", content="PLAN", kind="final"))["ok"] is True
    reader = group_message(text="lee el documento equipo.md", user_id=2)
    turn = service.workspace_for(reader)
    assert turn is not None
    assert (await call(turn, "read_document", name="equipo.md"))["content"] == "PLAN"
    assert await service._store.list(WorkspaceScope(ChatKind.GROUP, -100)) == (DocumentRef("equipo.md"),)


async def test_group_not_in_workspace_ids_denied(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    service = WorkspaceService(FileDocumentStore(root), frozenset(), frozenset({-100, -200}), frozenset({-100}))
    outside = group_message(text="lee documento a.md", chat_id=-200, user_id=1)
    assert service._scope(outside) is None
    assert service.workspace_for(outside) is None
    assert await service.attachment(group_message(text="envíame a.md", chat_id=-200, user_id=1)) is None


async def test_group_read_request_cannot_be_escalated_to_write_by_document_content(service):
    actor = group_message(text="lee el documento plan.md", user_id=1)
    turn = service.workspace_for(actor)
    assert turn is not None
    assert (await call(turn, "create_document", name="injected.md", content="x", kind="final"))["error"] == "access_denied"
    assert (await call(turn, "propose_edit", name="plan.md", content="x"))["error"] == "access_denied"
    assert await service._store.list(WorkspaceScope(ChatKind.GROUP, -100)) == ()


async def test_read_cannot_expose_unrequested_document_to_provider(service):
    scope = WorkspaceScope(ChatKind.PRIVATE, 1)
    await service._store.create(scope, DocumentRef("plan.md"), "PLAN")
    await service._store.create(scope, DocumentRef("privado.md"), "PRIVADO")
    turn = service.workspace_for(message("resume plan.md"))
    assert (await call(turn, "read_document", name="privado.md"))["error"] == "access_denied"
    assert (await call(turn, "read_document", name="plan.md"))["content"] == "PLAN"
    assert (await call(service.workspace_for(message("resume mis documentos")), "read_document", name="privado.md"))["error"] == "needs_clarification"


async def test_final_result_is_not_silently_written_to_temp(service):
    turn = service.workspace_for(message("guarda el documento final plan.md"))
    result = await call(turn, "create_document", name="plan.md", content="FINAL", kind="temp")
    assert result["name"] == "plan.md"
    assert (await service._store.read(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("plan.md")))[0] == "FINAL"


@pytest.mark.parametrize("text", ["hola", "gracias", "¿cómo estás?", "lee esto",
                                     "documento", "confirma TOKEN"])
def test_group_tools_reject_non_document_requests(service, text):
    assert service.workspace_for(group_message(text=text)) is None


@pytest.mark.parametrize("text", ["lista los documentos", "¿Cuáles son los archivos?",
                                     "dame el documento plan.md", "crea un documento",
                                     "edita plan.md", "read file plan.md"])
def test_group_tools_accept_explicit_natural_requests(service, text):
    assert service.workspace_for(group_message(text=text)) is not None


async def test_create_read_and_limit_are_isolated(service):
    turn = service.workspace_for(message("x"))
    assert (await call(turn, "create_document", name="a.md", content="secreto"))["ok"] is True
    assert "temp/a.md" in turn.notices[-1]
    assert (await call(turn, "create_document", name="a.md", content="otra"))["error"] == "already_exists"
    assert (await call(turn, "read_document", name="temp/a.md"))["content"] == "secreto"
    assert (await call(turn, "list_documents"))["documents"] == ["temp/a.md"]
    assert "temp/a.md" in turn.notices[-1]
    assert (await call(turn, "read_document", name="temp/a.md"))["ok"] is True
    assert (await call(turn, "read_document", name="temp/a.md"))["error"] == "operation_limit"
    other = service.workspace_for(message("x", user=2, chat=2))
    assert (await call(other, "list_documents"))["documents"] == []
    group = service.workspace_for(group_message(text="lista documentos"))
    assert (await call(group, "list_documents"))["documents"] == []


async def test_proposal_confirmation_conflict_and_reset(service):
    turn = service.workspace_for(message("x"))
    await call(turn, "create_document", name="plan.md", content="original", kind="final")
    proposal = await call(turn, "propose_edit", name="plan.md", content="nuevo")
    assert proposal["ok"] is True
    assert "nuevo" in turn.notices[-1] and "original" in turn.notices[-1]
    code = proposal["code"]
    service.preview_delivery(message("x"), "\n".join(turn.notices), True)
    assert "no se aplicó" in (await service.confirm(message(f"confirma {code}", user=2, chat=1)) or "")
    assert "no se aplicó" in (await service.confirm(message("confirma ERROR")) or "")
    assert "actualizado" in (await service.confirm(message(f"confirma {code}")) or "")
    assert await service.confirm(message(f"confirma {code}")) != "actualizado"
    assert (await call(service.workspace_for(message("x")), "read_document", name="plan.md"))["content"] == "nuevo"
    fresh = service.workspace_for(message("x"))
    code = await delivered_proposal(service, message("edita plan.md"), "plan.md", "propuesto")
    await service._store.replace_if_version(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("plan.md"), "externo", (await service._store.read(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("plan.md")))[1])
    assert "cambió" in (await service.confirm(message(f"confirma {code}")) or "")
    code = await delivered_proposal(service, message("edita plan.md"), "plan.md", "otro")
    service.reset(message("/reset"))
    assert "no se aplicó" in (await service.confirm(message(f"confirma {code}")) or "")


async def test_cancel_and_expiry(service, monkeypatch):
    await call(service.workspace_for(message("x")), "create_document", name="a.md", content="a", kind="final")
    code = await delivered_proposal(service, message("edita a.md"), "a.md", "b")
    assert "cancelada" in (await service.confirm(message("cancela")) or "")
    assert "no se aplicó" in (await service.confirm(message(f"confirma {code}")) or "")
    code = await delivered_proposal(service, message("edita a.md"), "a.md", "c")
    import oraculo.application.workspace as workspace_module

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(timezone.utc) + timedelta(minutes=11)

    monkeypatch.setattr(workspace_module, "datetime", Later)
    assert "venció" in (await service.confirm(message(f"confirma {code}")) or "")


async def test_attachment_requires_explicit_single_valid_name(service):
    await call(service.workspace_for(message("x")), "create_document", name="a.md", content="hola", kind="final")
    assert await service.attachment(message("resume a.md")) is None
    assert await service.attachment(message("envíame a.md y b.md")) is None
    assert await service.attachment(message("adjunta ../a.md")) is None
    assert await service.attachment(message("envíame a.md.pdf")) is None
    assert await service.attachment(message("adjunta a.md y b.pdf")) is None
    assert await service.attachment(message("envíame a.md")) == ("a.md", b"hola")
    assert await service.attachment(group_message(text="envíame a.md", user_id=2)) is None
    assert await service.attachment(message("No adjunta a.md; solo dime qué contiene")) is None


@pytest.mark.parametrize("text", ["envíame", "envíame el archivo", "envíame a.md y b.md",
                                      "envíame a.md.pdf", "adjunta ../a.md", "adjunta a.md y b.pdf"])
async def test_ambiguous_attachment_asks_for_unique_name_without_document_io(service, monkeypatch, text):
    async def forbidden(*args):
        pytest.fail("An ambiguous request must not access documents")

    monkeypatch.setattr(service._store, "attachment", forbidden)
    monkeypatch.setattr(service._store, "read", forbidden)
    assert ".md" in (service.attachment_clarification(message(text)) or "")
    assert ".txt" in (service.attachment_clarification(message(text)) or "")
    assert await service.attachment(message(text)) is None


async def test_group_nonmember_attachment_does_not_disclose_names(service):
    request = group_message(text="envíame secreto.md", user_id=2)
    assert service.attachment_clarification(request) is not None
    assert "secreto.md" not in service.attachment_clarification(request)
    assert await service.attachment(request) is None


async def test_private_long_read_is_available_to_ai_and_offers_attachment(service):
    actor = message("lee largo.md")
    await service._store.create(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("largo.md"), "X" * 1901)
    turn = service.workspace_for(actor)
    result = await call(turn, "read_document", name="largo.md")
    assert result["content"] == "X" * 1901
    assert "X" * 1901 in " ".join(turn.notices)
    assert "envíame largo.md" in " ".join(turn.notices)


async def test_group_long_read_does_not_offer_automatic_attachment(service):
    actor = group_message(text="lee largo.md")
    await service._store.create(WorkspaceScope(ChatKind.GROUP, -100), DocumentRef("largo.md"), "X" * 1901)
    turn = service.workspace_for(actor)
    result = await call(turn, "read_document", name="largo.md")
    assert result["content"] == "X" * 1901
    assert "X" * 1901 not in " ".join(turn.notices)
    assert "acota" in " ".join(turn.notices).lower()
    assert "envíame" not in " ".join(turn.notices).lower()
    assert "visible para el grupo" in " ".join(turn.notices).lower()


async def test_concurrent_confirm_is_one_use(service):
    await call(service.workspace_for(message("x")), "create_document", name="a.md", content="a", kind="final")
    code = await delivered_proposal(service, message("edita a.md"), "a.md", "b")
    results = await asyncio.gather(*(service.confirm(message(f"confirma {code}")) for _ in range(2)))
    assert sum("actualizado" in (result or "") for result in results) == 1


async def test_group_reset_invalidates_other_members_proposals(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    service = WorkspaceService(FileDocumentStore(root), frozenset(), frozenset({-100}), frozenset({-100}))
    first = group_message(text="edita a.md", user_id=1)
    await call(service.workspace_for(first), "create_document", name="a.md", content="old", kind="final")
    code = (await call(service.workspace_for(first), "propose_edit", name="a.md", content="new"))["code"]
    service.preview_delivery(first, f"confirma {code}", True)
    service.reset(group_message(user_id=2))
    assert "no se aplicó" in (await service.confirm(group_message(text=f"confirma {code}")) or "")


async def test_preview_must_be_delivered_before_confirmation_and_failed_delivery_revokes_it(service):
    actor = message("edita a.md")
    await call(service.workspace_for(actor), "create_document", name="a.md", content="old", kind="final")
    turn = service.workspace_for(actor)
    code = (await call(turn, "propose_edit", name="a.md", content="new"))["code"]
    preview = "\n".join(turn.notices)
    assert "no se aplicó" in (await service.confirm(message(f"confirma {code}")) or "")
    service.preview_delivery(actor, preview, False)
    service.preview_delivery(actor, preview, True)
    assert "no se aplicó" in (await service.confirm(message(f"confirma {code}")) or "")
    turn = service.workspace_for(actor)
    code = (await call(turn, "propose_edit", name="a.md", content="new"))["code"]
    service.preview_delivery(actor, "\n".join(turn.notices), True)
    assert "actualizado" in (await service.confirm(message(f"confirma {code}")) or "")


async def test_durability_unknown_never_claims_write_was_not_applied(service, monkeypatch):
    actor = message("edita a.md")
    store = service._store
    await store.create(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("a.md"), "old")
    original_create = store.create
    original_replace = store.replace_if_version

    async def uncertain_create(scope, ref, content):
        await original_create(scope, ref, content)
        raise WorkspaceError("durability_unknown")

    monkeypatch.setattr(store, "create", uncertain_create)
    turn = service.workspace_for(actor)
    created = await call(turn, "create_document", name="b.md", content="new", kind="final")
    assert created == {"error": "durability_unknown"}
    assert "pudo crearse" in turn.notices[-1]
    monkeypatch.setattr(store, "create", original_create)

    async def uncertain_replace(scope, ref, content, version):
        await original_replace(scope, ref, content, version)
        raise WorkspaceError("durability_unknown")

    monkeypatch.setattr(store, "replace_if_version", uncertain_replace)
    code = await delivered_proposal(service, actor, "a.md", "updated")
    result = await service.confirm(message(f"confirma {code}"))
    assert "pudo actualizarse" in (result or "")
    assert "No se aplicó" not in (result or "")
