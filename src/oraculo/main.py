"""Composition root: carga configuración, inyecta dependencias y arranca."""

from __future__ import annotations

import logging
import os
import sys
import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from telegram.ext import Application

from oraculo.application.attachments import AttachmentService
from oraculo.application.conversation import ConversationService
from oraculo.application.pi_mode import PiModeService, ROOT_PROJECT_LABEL
from oraculo.domain.pi import PiProject
from oraculo.application.sessions import SessionManager
from oraculo.application.skills import SkillService
from oraculo.domain.errors import ConfigError
from oraculo.domain.models import AppConfig, SkillCatalogSnapshot
from oraculo.infrastructure.access import EnvAccessPolicy
from oraculo.infrastructure.audio_transcription import FasterWhisperTranscriber
from oraculo.infrastructure.config import load_config
from oraculo.infrastructure.deepseek import DeepSeekClient
from oraculo.infrastructure.mimo import MimoClient
from oraculo.infrastructure.codex import CodexClient
from oraculo.infrastructure.codex_auth import AuthError, CodexBrowserLogin, OAuthCredentialStore
from oraculo.infrastructure.logging import configure_logging
from oraculo.infrastructure.phrases import RandomPhraseProvider
from oraculo.infrastructure.pi_projects import FilesystemProjectStore
from oraculo.infrastructure.pi_rpc import PiRpcClient
from oraculo.infrastructure.sessions import InMemorySessionRepository
from oraculo.infrastructure.skill_catalog import FileSkillCatalog
from oraculo.infrastructure.temp_attachments import TempAttachmentStore
from oraculo.infrastructure.text_extractor import TextExtractor
from oraculo.infrastructure.workspace import FileDocumentStore
from oraculo.domain.workspace import WorkspaceError
from oraculo.telegram.bot import build_application
from oraculo.telegram.session_locks import SessionUpdateLockPool

logger = logging.getLogger(__name__)

ATTACHMENT_TEXT_MAX_CHARS = 20_000


def _binary_supported(provider: str) -> Callable[[str, str], bool]:
    """Tipos que viajan en binario al proveedor activo (Req 2.2/2.3)."""
    if provider == "mimo":
        # Sin Files API: solo imágenes por data URL; el resto se extrae a texto.
        return lambda name, mime: mime.startswith("image/")
    return lambda name, mime: True


def _new_http_client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout_s)


def _credential_store(path: str) -> OAuthCredentialStore:
    location = Path(path) if path else Path.home() / ".local" / "share" / "oraculo" / "oauth.json"
    try:
        return OAuthCredentialStore(location)
    except AuthError as exc:
        raise ConfigError("GPT_CREDENTIAL_PATH", str(exc)) from None


def create_application(config: AppConfig) -> Application:
    """Construye una ejecución completa y conecta sus recursos al lifecycle."""
    workspace_service = None
    if config.workspace_enabled:
        from oraculo.application.workspace import WorkspaceService

        try:
            document_store = FileDocumentStore(Path(config.workspace_dir))
        except WorkspaceError:
            raise ConfigError("WORKSPACE_DIR", "raíz de workspace inválida o insegura") from None
        workspace_service = WorkspaceService(
            store=document_store,
            allowed_user_ids=config.allowed_user_ids,
            allowed_chat_ids=config.allowed_chat_ids,
            workspace_group_ids=config.workspace_group_ids,
        )
    if config.ai_provider == "gpt":
        store = _credential_store(config.gpt_credential_path)
        try:
            if not store.available_for_startup():
                raise ConfigError("AI_PROVIDER", "inicia sesión localmente con auth login")
        except AuthError as exc:
            raise ConfigError("AI_PROVIDER", str(exc)) from None
        http_client = _new_http_client(config.gpt_timeout_s)
        ai_client = CodexClient(
            store=store, http_client=http_client,
            system_prompt=config.deepseek_system_prompt,
            model=config.gpt_model, timeout_s=config.gpt_timeout_s,
        )
    elif config.ai_provider == "deepseek":
        http_client = _new_http_client(config.deepseek_timeout_s)
        ai_client = DeepSeekClient(
            api_key=config.deepseek_api_key,
            model=config.deepseek_model,
            timeout_s=config.deepseek_timeout_s,
            max_tokens=config.deepseek_max_tokens,
            system_prompt=config.deepseek_system_prompt,
            thinking=config.deepseek_thinking,
            reasoning_effort=config.deepseek_reasoning_effort,
            http_client=http_client,
        )
    else:
        http_client = _new_http_client(config.mimo_timeout_s)
        ai_client = MimoClient(
            api_key=config.mimo_api_key,
            model=config.mimo_model,
            timeout_s=config.mimo_timeout_s,
            max_tokens=config.mimo_max_tokens,
            system_prompt=config.deepseek_system_prompt,
            http_client=http_client,
            base_url=config.mimo_base_url,
        )
    audio_transcriber = FasterWhisperTranscriber()
    if config.skills_enabled:
        snapshot = FileSkillCatalog(
            config.skills_dir, max_chars=config.skill_max_chars
        ).snapshot
    else:
        snapshot = SkillCatalogSnapshot()
        logger.info("Skill catalog disabled loaded=0 skipped=0")
    skill_service = SkillService(
        snapshot,
        autoload_enabled=config.skills_autoload,
        catalog_max_chars=config.skills_catalog_max_chars,
        max_context_chars=config.skills_max_context_chars,
        max_per_turn=config.skills_max_per_turn,
        max_rounds=config.skills_max_rounds,
    )
    access_policy = EnvAccessPolicy(
        allowed_user_ids=config.allowed_user_ids,
        allowed_chat_ids=config.allowed_chat_ids,
    )
    attachment_service = AttachmentService(
        store=TempAttachmentStore(
            root=Path(__file__).resolve().parents[2] / "temp" / "attachments"
        ),
        binary_supported=_binary_supported(config.ai_provider),
        clock=lambda: datetime.now(timezone.utc),
        extract_text=TextExtractor(max_chars=ATTACHMENT_TEXT_MAX_CHARS).extract,
        idle_minutes=config.session_idle_minutes,
    )
    phrase_provider = RandomPhraseProvider(
        rejection=config.rejection_phrases,
        only_text=config.only_text_phrases,
        empty_mention=config.empty_mention_phrases,
        new_session=config.new_session_phrases,
        skill_limit=config.skills_limit_phrase,
    )
    sessions = SessionManager(
        repository=InMemorySessionRepository(),
        max_turns=config.session_max_turns,
        idle_minutes=config.session_idle_minutes,
        clock=lambda: datetime.now(timezone.utc),
    )
    service = ConversationService(
        ai_client=ai_client,
        access_policy=access_policy,
        phrase_provider=phrase_provider,
        sessions=sessions,
        skill_service=skill_service,
        workspace_service=workspace_service,
        attachment_service=attachment_service,
        audio_transcriber=audio_transcriber,
    )

    pi_mode_service = None
    if config.pi_enabled:
        project_root = Path(__file__).resolve().parents[2]
        pi_workspace = Path(config.pi_workspace_dir)
        if not pi_workspace.is_absolute():
            pi_workspace = project_root / pi_workspace
        pi_workspace.mkdir(parents=True, exist_ok=True)
        # La raíz contiene los proyectos; cada sesión elige uno como `cwd`.
        pi_projects = FilesystemProjectStore(root=pi_workspace)

        async def pi_agent_factory(
            user_id: int, cwd: str, session_file: str | None = None
        ) -> PiRpcClient:
            # `cwd` ya viene resuelto y confinado por `FilesystemProjectStore`.
            # `session_file` retoma la sesión anterior de Pi; no se registra (Req 2.3.3).
            del user_id
            logger.info("Lanzando Pi RPC en %s reanudar=%s", cwd, session_file is not None)
            client = await PiRpcClient.spawn(cwd=cwd, session_file=session_file)
            logger.info("Pi RPC lanzado correctamente")
            return client

        # La raíz del proyecto es una opción extra solo para usuarios root.
        root_project = (
            PiProject(name=ROOT_PROJECT_LABEL, path=str(project_root))
            if config.pi_root_user_ids else None
        )
        pi_mode_service = PiModeService(
            agent_factory=pi_agent_factory,
            project_store=pi_projects,
            root_project=root_project,
            root_user_ids=config.pi_root_user_ids,
            audio_transcriber=audio_transcriber,
        )
        logger.info(
            "Modo Pi habilitado workspace=%s root_users=%d",
            pi_workspace,
            len(config.pi_root_user_ids),
        )

    if not config.allowed_user_ids:
        logger.warning("ALLOWED_USER_IDS está vacío: nadie podrá usar el bot en privado")
    if not config.allowed_chat_ids:
        logger.warning("ALLOWED_CHAT_IDS está vacío: el bot no responderá en grupos")
    if not sessions.enabled:
        logger.info("Memoria conversacional desactivada (SESSION_MAX_TURNS=0)")

    async def startup_attachments(application: Application) -> None:
        del application
        await attachment_service.startup()

    async def close_http_client(application: Application) -> None:
        del application
        await ai_client.aclose()

    return build_application(
        config.telegram_bot_token,
        service,
        pi_mode_service=pi_mode_service,
        pi_stream_edit_interval_s=config.pi_stream_edit_interval_s,
        max_concurrent_updates=config.max_concurrent_updates,
        session_update_locks=SessionUpdateLockPool(),
        post_shutdown=close_http_client,
        post_init=startup_attachments,
    )


async def _login() -> None:
    """Comando local: nunca atraviesa el handler ni las sesiones de Telegram."""
    store = _credential_store(os.environ.get("GPT_CREDENTIAL_PATH", ""))
    async with httpx.AsyncClient() as client:
        login = CodexBrowserLogin(store=store, http_client=client)
        await login.login(lambda url: print(f"Abre esta URL en tu navegador local:\n{url}"))
    print("Cuenta conectada; las credenciales se almacenaron fuera del repositorio.")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = argv if argv is not None else []
    if args:
        if args != ["auth", "login"]:
            logger.error("Uso: python -m oraculo.main [auth login]")
            raise SystemExit(2)
        try:
            asyncio.run(_login())
        except (AuthError, ConfigError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from None
        except KeyboardInterrupt:
            logger.info("Login cancelado")
            raise SystemExit(130) from None
        return
    try:
        config = load_config()
        application = create_application(config)
    except ConfigError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    logger.info("Oraculo iniciado en modo polling")
    application.run_polling()


if __name__ == "__main__":
    main(sys.argv[1:])
