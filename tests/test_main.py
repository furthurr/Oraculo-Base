"""Tests del composition root y su ciclo de vida."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import time

import httpx
import pytest
from _pytest.monkeypatch import MonkeyPatch
from telegram.ext import Application

from oraculo.infrastructure.config import load_config
from oraculo.infrastructure.codex_auth import OAuthCredential, OAuthCredentialStore
from oraculo.domain.models import AppConfig
from oraculo.main import create_application, main
from oraculo.application.conversation import ConversationService
from oraculo.application.pi_mode import PiModeService
from oraculo.domain.errors import ConfigError
from oraculo.domain.workspace import WorkspaceError
from oraculo.domain.workspace import DocumentRef, WorkspaceScope
from oraculo.domain.models import ChatKind
from oraculo.infrastructure.audio_transcription import FasterWhisperTranscriber


def config_for(skills_dir: Path, *, enabled: bool = True) -> AppConfig:
    return load_config(
        {
            "TELEGRAM_BOT_TOKEN": "123:abc",
            "AI_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "sk-test",
            "SKILLS_DIR": str(skills_dir),
            "SKILLS_ENABLED": "enabled" if enabled else "disabled",
        }
    )


async def test_arranca_con_directorio_de_skills_inexistente(tmp_path: Path):
    application = create_application(config_for(tmp_path / "missing"))

    assert isinstance(application, Application)
    assert application.post_shutdown is not None
    await application.post_shutdown(application)


async def test_create_application_propaga_el_limite_de_concurrencia() -> None:
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "AI_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "sk-test",
        "SKILLS_ENABLED": "disabled",
        "MAX_CONCURRENT_UPDATES": "3",
    })
    application = create_application(config)

    try:
        assert application.concurrent_updates == 3
    finally:
        assert application.post_shutdown is not None
        await application.post_shutdown(application)


async def test_arranca_omitiendo_skills_invalidas(tmp_path: Path):
    skill_dir = tmp_path / "invalid"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("sin frontmatter", encoding="utf-8")

    application = create_application(config_for(tmp_path))

    assert isinstance(application, Application)
    assert application.post_shutdown is not None
    await application.post_shutdown(application)


async def test_post_shutdown_cierra_el_unico_cliente_http(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    clients: list[httpx.AsyncClient] = []
    async_client_type = httpx.AsyncClient

    def client_factory(timeout: float) -> httpx.AsyncClient:
        client = async_client_type(timeout=timeout)
        clients.append(client)
        return client

    monkeypatch.setattr("oraculo.main._new_http_client", client_factory)
    application = create_application(config_for(tmp_path / "missing"))

    assert len(clients) == 1
    assert application.post_shutdown is not None
    await application.post_shutdown(application)

    assert clients[0].is_closed


async def test_transcriptor_local_se_inyecta_sin_clave_ni_cliente_http_stt(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "AI_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "sk-test",
        "SKILLS_ENABLED": "disabled",
    })
    clients: list[httpx.AsyncClient] = []
    async_client_type = httpx.AsyncClient

    def client_factory(timeout: float) -> httpx.AsyncClient:
        client = async_client_type(timeout=timeout)
        clients.append(client)
        return client

    monkeypatch.setattr("oraculo.main._new_http_client", client_factory)
    application = create_application(config)

    try:
        assert len(clients) == 1
        assert [client.timeout.read for client in clients] == [30.0]
        message_handler = application.handlers[0][1]
        service = message_handler.callback._service
        assert isinstance(service, ConversationService)
        assert isinstance(service._audio_transcriber, FasterWhisperTranscriber)
        assert service._audio_transcriber._model is None
    finally:
        assert application.post_shutdown is not None
        await application.post_shutdown(application)

    assert clients[0].is_closed


async def test_transcriptor_se_comparte_con_el_modo_pi(
    tmp_path: Path,
) -> None:
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "AI_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "sk-test",
        "PI_ENABLED": "enabled",
        "PI_WORKSPACE_DIR": str(tmp_path / "pi-workspace"),
        "SKILLS_ENABLED": "disabled",
    })
    application = create_application(config)

    try:
        message_handler = application.handlers[0][1]
        callback = message_handler.callback
        service = callback._service
        pi_mode_service = callback._pi_mode_service

        assert isinstance(service, ConversationService)
        assert isinstance(pi_mode_service, PiModeService)
        assert isinstance(service._audio_transcriber, FasterWhisperTranscriber)
        assert pi_mode_service._audio_transcriber is service._audio_transcriber
        assert service._audio_transcriber._model is None
    finally:
        assert application.post_shutdown is not None
        await application.post_shutdown(application)


async def test_skills_disabled_no_lee_el_directorio(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def fail_if_constructed(root: str, *, max_chars: int) -> None:
        del root, max_chars
        raise AssertionError("no debe leer el catálogo")

    monkeypatch.setattr("oraculo.main.FileSkillCatalog", fail_if_constructed)

    application = create_application(config_for(tmp_path, enabled=False))

    assert isinstance(application, Application)
    assert application.post_shutdown is not None
    await application.post_shutdown(application)


async def test_gpt_con_credencial_construye_y_cierra_su_cliente(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    store = OAuthCredentialStore(tmp_path / "private" / "oauth.json")
    await store.save(OAuthCredential("access-test", "refresh-test", int(time.time()) + 3600, "account-test"))
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "gpt",
        "GPT_CREDENTIAL_PATH": str(store.path), "SKILLS_ENABLED": "disabled",
    })
    clients: list[httpx.AsyncClient] = []

    def create_client(timeout: float) -> httpx.AsyncClient:
        client = httpx.AsyncClient(timeout=timeout)
        clients.append(client)
        return client

    monkeypatch.setattr("oraculo.main._new_http_client", create_client)
    application = create_application(config)
    assert isinstance(application, Application)
    assert len(clients) == 1
    assert application.post_shutdown is not None
    await application.post_shutdown(application)
    assert clients[0].is_closed


def test_gpt_sin_credencial_impide_arrancar_sin_fallback(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr("oraculo.main.configure_logging", lambda: None)
    monkeypatch.setattr(
        "oraculo.main.load_config",
        lambda: load_config(
            {
                "TELEGRAM_BOT_TOKEN": "123:abc",
                "AI_PROVIDER": "gpt",
                "GPT_CREDENTIAL_PATH": str(tmp_path / "private" / "oauth.json"),
            }
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1


async def test_mimo_con_api_key_construye_y_cierra_su_cliente(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "mimo",
        "MIMO_API_KEY": "mk-test", "MIMO_MODEL": "xiaomi/mimo-v2.6-pro",
        "SKILLS_ENABLED": "disabled",
    })
    clients: list[httpx.AsyncClient] = []

    def create_client(timeout: float) -> httpx.AsyncClient:
        client = httpx.AsyncClient(timeout=timeout)
        clients.append(client)
        return client

    monkeypatch.setattr("oraculo.main._new_http_client", create_client)
    application = create_application(config)

    assert isinstance(application, Application)
    assert len(clients) == 1
    assert application.post_shutdown is not None
    await application.post_shutdown(application)
    assert clients[0].is_closed


async def test_sin_ai_provider_por_defecto_construye_mimo(
    monkeypatch: MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class FakeMimoClient:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)
            self._http = kwargs["http_client"]

        async def aclose(self) -> None:
            assert isinstance(self._http, httpx.AsyncClient)
            await self._http.aclose()

    monkeypatch.setattr("oraculo.main.MimoClient", FakeMimoClient)
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc", "MIMO_API_KEY": "mk-test",
        "SKILLS_ENABLED": "disabled",
    })
    application = create_application(config)

    assert config.ai_provider == "mimo"
    assert len(created) == 1
    assert created[0]["api_key"] == "mk-test"
    assert created[0]["model"] == "xiaomi/mimo-v2.6-flash"
    assert created[0]["base_url"] == "https://api.xiaomimimo.com/v1"
    assert application.post_shutdown is not None
    await application.post_shutdown(application)


def test_comando_auth_login_no_arranca_telegram(monkeypatch: MonkeyPatch) -> None:
    called: list[str] = []

    async def fake_login() -> None:
        called.append("auth")

    monkeypatch.setattr("oraculo.main._login", fake_login)
    monkeypatch.setattr("oraculo.main.configure_logging", lambda: None)
    monkeypatch.setattr("oraculo.main.load_config", lambda: pytest.fail("No debe leer token de Telegram"))
    main(["auth", "login"])
    assert called == ["auth"]


async def test_workspace_desactivado_no_construye_store(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("oraculo.main.FileDocumentStore", lambda root: pytest.fail("store apagado"))
    application = create_application(config_for(tmp_path, enabled=False))
    assert application.post_shutdown is not None
    await application.post_shutdown(application)


async def test_workspace_habilitado_inyecta_grupos(tmp_project: Path, monkeypatch: MonkeyPatch) -> None:
    root = tmp_project / "workspace"
    root.mkdir(mode=0o700)
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-test",
                          "SKILLS_ENABLED": "disabled", "WORKSPACE_ENABLED": "enabled",
                          "WORKSPACE_DIR": str(root), "ALLOWED_CHAT_IDS": "-100",
                          "WORKSPACE_GROUP_IDS": "-100"})
    observed: list[Path] = []
    from oraculo.infrastructure.workspace import FileDocumentStore

    def store_factory(path: Path) -> FileDocumentStore:
        observed.append(path)
        return FileDocumentStore(path)

    monkeypatch.setattr("oraculo.main.FileDocumentStore", store_factory)
    application = create_application(config)
    try:
        assert observed == [root]
        handler = application.handlers[0][1]
        assert isinstance(handler.callback._service, ConversationService)
        assert handler.callback._service._workspace_service._workspace_group_ids == frozenset({-100})
        assert callable(handler.callback._service.handle_attachment)
    finally:
        assert application.post_shutdown is not None
        await application.post_shutdown(application)


def test_workspace_error_al_iniciar_no_expone_rutas(tmp_project: Path, monkeypatch: MonkeyPatch) -> None:
    root = tmp_project / "workspace"
    root.mkdir(mode=0o700)
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-test",
                          "SKILLS_ENABLED": "disabled", "WORKSPACE_ENABLED": "enabled",
                          "WORKSPACE_DIR": str(root)})
    def broken_store(path: Path) -> None:
        raise WorkspaceError("invalid_root")
    monkeypatch.setattr("oraculo.main.FileDocumentStore", broken_store)
    with pytest.raises(ConfigError) as excinfo:
        create_application(config)
    assert excinfo.value.variable == "WORKSPACE_DIR"
    assert str(root) not in str(excinfo.value)


async def test_integracion_adjunto_privado_habilitado(tmp_project: Path) -> None:
    root = tmp_project / "workspace"
    root.mkdir(mode=0o700)
    from oraculo.infrastructure.workspace import FileDocumentStore
    await FileDocumentStore(root).create(WorkspaceScope(ChatKind.PRIVATE, 1), DocumentRef("plan.md"), "informe")
    config = load_config({"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-test",
                          "SKILLS_ENABLED": "disabled", "WORKSPACE_ENABLED": "enabled",
                          "WORKSPACE_DIR": str(root), "ALLOWED_USER_IDS": "1"})
    application = create_application(config)

    class Message:
        text = "envíame plan.md"
        entities = ()
        caption_entities = ()
        reply_to_message = None
        message_thread_id = None
        documents: list[tuple[str, bytes]] = []
        replies: list[str] = []

        async def reply_document(self, document: BytesIO, *, filename: str) -> None:
            self.documents.append((filename, document.read()))

        async def reply_text(self, text: str) -> None:
            self.replies.append(text)

    message = Message()
    update = SimpleNamespace(effective_message=message,
                             effective_chat=SimpleNamespace(type="private", id=1),
                             effective_user=SimpleNamespace(id=1, is_bot=False, first_name="Ana", username=None))
    context = SimpleNamespace(bot=SimpleNamespace(id=42, username="oraculo_bot"))
    try:
        await application.handlers[0][1].callback(update, context)
        assert message.documents == [("plan.md", b"informe")]
        assert message.replies == ["Archivo entregado: plan.md."]
    finally:
        assert application.post_shutdown is not None
        await application.post_shutdown(application)


def test_binary_supported_por_proveedor() -> None:
    from oraculo.main import _binary_supported

    mimo = _binary_supported("mimo")
    assert mimo("foto.jpg", "image/jpeg") is True
    assert mimo("datos.zip", "application/zip") is False
    deepseek = _binary_supported("deepseek")
    assert deepseek("datos.zip", "application/zip") is True
    gpt = _binary_supported("gpt")
    assert gpt("datos.zip", "application/zip") is True


async def test_wiring_de_adjuntos_construye_servicio_para_mimo(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    import oraculo.main as main_module
    from oraculo.infrastructure.temp_attachments import TempAttachmentStore

    roots: list[Path] = []

    def store_factory(root: Path) -> object:
        roots.append(root)
        return TempAttachmentStore(root=root)

    monkeypatch.setattr("oraculo.main.TempAttachmentStore", store_factory)
    config = load_config({
        "TELEGRAM_BOT_TOKEN": "123:abc", "MIMO_API_KEY": "mk-test",
        "SKILLS_ENABLED": "disabled",
    })
    application = create_application(config)

    handler = application.handlers[0][1]
    attachment_service = handler.callback._service._attachment_service
    assert attachment_service is not None
    expected = Path(main_module.__file__).resolve().parents[2] / "temp" / "attachments"
    assert roots == [expected]
    assert attachment_service._binary_supported("foto.jpg", "image/jpeg") is True
    assert attachment_service._binary_supported("datos.zip", "application/zip") is False
    assert application.post_shutdown is not None
    await application.post_shutdown(application)


async def test_post_init_limpia_los_adjuntos_al_arrancar(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    started: list[bool] = []

    class FakeAttachmentService:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def startup(self) -> None:
            started.append(True)

    monkeypatch.setattr("oraculo.main.AttachmentService", FakeAttachmentService)
    application = create_application(config_for(tmp_path / "missing"))

    async def skip_command_registration(bot: object, commands: object) -> None:
        del bot, commands

    monkeypatch.setattr(
        type(application.bot), "set_my_commands", skip_command_registration
    )
    assert application.post_init is not None
    await application.post_init(application)
    assert started == [True]
    assert application.post_shutdown is not None
    await application.post_shutdown(application)
