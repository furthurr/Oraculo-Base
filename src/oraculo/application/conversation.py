"""Caso de uso: decidir y producir la respuesta a un mensaje entrante."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from oraculo.application.attachments import AttachmentService, NextAction
from oraculo.application.audio import format_voice_prompt, transcribe_voice
from oraculo.application.budget import (
    DEFAULT_REQUEST_BUDGET_BYTES,
    payload_size,
)
from oraculo.application.workspace import WorkspaceService
from oraculo.application.sessions import SessionManager, with_author
from oraculo.application.skills import SkillService
from oraculo.application.pi_mode import PI_PREFIX
from oraculo.domain.attachments import AttachmentMode, AttachmentPayload, IncomingAttachment
from oraculo.domain.errors import AIError, AudioTranscriptionError, SkillLimitError
from oraculo.domain.models import (
    AttachmentKey,
    ChatKind,
    ChatMessage,
    IncomingMessage,
    MessageRole,
    SessionKey,
)
from oraculo.domain.pi import PiAccess
from oraculo.domain.ports import AudioTranscriber, AIClient, AccessPolicy, PhraseProvider

logger = logging.getLogger(__name__)
COMPACTION_NOTICE = "He resumido el contexto de esta sesión para poder continuar."
PI_GROUP_NOTICE = f"{PI_PREFIX} El modo agente solo está disponible por mensaje directo."


class ConversationService:
    """Orquesta acceso, sesiones, rechazo y consulta a la IA según el chat."""

    def __init__(
        self,
        *,
        ai_client: AIClient,
        access_policy: AccessPolicy,
        phrase_provider: PhraseProvider,
        sessions: SessionManager,
        skill_service: SkillService,
        workspace_service: WorkspaceService | None = None,
        attachment_service: AttachmentService | None = None,
        audio_transcriber: AudioTranscriber | None = None,
        attachment_budget_bytes: int = DEFAULT_REQUEST_BUDGET_BYTES,
    ) -> None:
        self._ai_client = ai_client
        self._access_policy = access_policy
        self._phrase_provider = phrase_provider
        self._sessions = sessions
        self._skill_service = skill_service
        self._workspace_service = workspace_service
        self._attachment_service = attachment_service
        self._audio_transcriber = audio_transcriber
        self._attachment_budget_bytes = attachment_budget_bytes

    def pi_access(self, message: IncomingMessage) -> PiAccess:
        """Decide si un mensaje puede llegar al modo Pi (solo privado y autorizado).

        En grupos autorizados devuelve un aviso; en grupos no autorizados u
        otros chats, silencio; en privado no autorizado, la frase de rechazo.
        """
        if message.chat_type is ChatKind.PRIVATE:
            if self._access_policy.is_user_allowed(message.user_id):
                return PiAccess(allowed=True, reply=None)
            logger.info("Modo Pi rechazado en privado user_id=%s", message.user_id)
            return PiAccess(allowed=False, reply=self._phrase_provider.rejection_phrase())
        if (
            message.chat_type is ChatKind.GROUP
            and self._access_policy.is_chat_allowed(message.chat_id)
        ):
            return PiAccess(allowed=False, reply=PI_GROUP_NOTICE)
        return PiAccess(allowed=False, reply=None)

    async def handle(self, message: IncomingMessage) -> str | None:
        """Devuelve el texto a responder, o `None` si el mensaje se ignora."""
        if message.chat_type is ChatKind.PRIVATE:
            return await self._handle_private(message)
        if message.chat_type is ChatKind.GROUP:
            return await self._handle_group(message)
        return None

    def _attachment_key(self, message: IncomingMessage) -> AttachmentKey | None:
        if message.chat_type not in (ChatKind.PRIVATE, ChatKind.GROUP):
            return None
        return AttachmentKey(
            kind=message.chat_type,
            chat_id=message.chat_id,
            user_id=message.user_id,
        )

    async def handle_attachments(
        self,
        message: IncomingMessage,
        items: Sequence[tuple[IncomingAttachment, bytes | None]],
    ) -> str | None:
        """Procesa una interacción con archivos adjuntos (Req 1.x–3.x)."""
        voice_items = tuple(item for item in items if item[0].kind == "voice")
        if self._attachment_service is None and not voice_items:
            return None
        if message.chat_type is ChatKind.PRIVATE:
            if not self._access_policy.is_user_allowed(message.user_id):
                logger.info(
                    "Adjuntos rechazados en privado user_id=%s", message.user_id,
                )
                return self._phrase_provider.rejection_phrase()
        elif message.chat_type is ChatKind.GROUP:
            if not self._access_policy.is_chat_allowed(message.chat_id) or not (
                message.mentions_bot or message.replies_to_bot
            ):
                return None
        else:
            return None
        key = self._attachment_key(message)
        if key is None:
            return None
        if self._attachment_service is not None:
            await self._attachment_service.expire_if_stale(key)
        if voice_items:
            if len(items) != 1:
                return "Envía la nota de voz separada de los demás archivos, por favor."
            return await self._handle_voice(message, *voice_items[0])
        if self._attachment_service is None:
            return None
        outcome = await self._attachment_service.intake(
            key, items, request_text=(message.text or "").strip() or None,
        )
        reply_parts = list(outcome.notices)
        if outcome.action is NextAction.ASK_INTENT:
            if outcome.has_valid_attachments:
                reply_parts.append(
                    "¿Procedo con los archivos válidos o prefieres adjuntar otros?"
                )
            else:
                reply_parts.append(
                    "Adjunta archivos compatibles para que pueda procesar la solicitud."
                )
            return "\n\n".join(reply_parts)
        if outcome.action is NextAction.ASK_INSTRUCTIONS:
            reply_parts.append("¿Qué quieres hacer con los archivos adjuntos?")
            return "\n\n".join(reply_parts)
        session_key = (
            SessionKey.private(message.user_id)
            if message.chat_type is ChatKind.PRIVATE
            else SessionKey.group(message.chat_id)
        )
        reply = await self._respond_to_text(message, session_key)
        if reply is None:
            return "\n\n".join(reply_parts) if reply_parts else None
        if reply_parts:
            return "\n\n".join(reply_parts) + "\n\n" + reply
        return reply

    async def _handle_voice(
        self,
        message: IncomingMessage,
        attachment: IncomingAttachment,
        data: bytes | None,
    ) -> str | None:
        """Transcribe una nota `voice` como el turno conversacional actual."""
        try:
            transcript = await transcribe_voice(
                attachment, data, transcriber=self._audio_transcriber,
            )
        except AudioTranscriptionError as exc:
            logger.warning(
                "No se pudo transcribir nota de voz user_id=%s chat_id=%s reason=%s",
                message.user_id, message.chat_id, exc.reason,
            )
            if exc.reason == "file_too_large":
                return "La nota de voz supera el máximo de 20 MB permitido."
            if exc.reason == "download_failed":
                return "No pude descargar la nota de voz; vuelve a enviarla."
            if exc.reason == "unsupported_format":
                return "No pude reconocer el formato de la nota de voz."
            if exc.reason == "empty_transcript":
                return "No detecté voz hablada en la nota de voz."
            return self._audio_error_reply(exc)
        prompt = format_voice_prompt(message.text, transcript)
        normalized = replace(message, text=prompt)
        session_key = (
            SessionKey.private(message.user_id)
            if message.chat_type is ChatKind.PRIVATE
            else SessionKey.group(message.chat_id)
        )
        return await self._respond_to_text(normalized, session_key)

    async def handle_attachment(self, message: IncomingMessage) -> tuple[str, bytes] | None:
        """Resuelve únicamente solicitudes explícitas en un chat conversable."""
        if self._workspace_service is None:
            return None
        if message.chat_type is ChatKind.PRIVATE:
            if not self._access_policy.is_user_allowed(message.user_id):
                return None
        elif message.chat_type is ChatKind.GROUP:
            if not self._access_policy.is_chat_allowed(message.chat_id) or not (
                message.mentions_bot or message.replies_to_bot
            ):
                return None
        else:
            return None
        return await self._workspace_service.attachment(message)

    def preview_delivery(self, message: IncomingMessage, reply: str, sent: bool) -> None:
        if self._workspace_service is not None:
            self._workspace_service.preview_delivery(message, reply, sent)

    async def handle_reset(self, message: IncomingMessage) -> str | None:
        """Atiende `/new` y `/reset`: descarta la sesión y confirma."""
        if message.chat_type is ChatKind.PRIVATE:
            if not self._access_policy.is_user_allowed(message.user_id):
                logger.info(
                    "Reinicio rechazado en privado user_id=%s chat_id=%s",
                    message.user_id,
                    message.chat_id,
                )
                return self._phrase_provider.rejection_phrase()
            key = SessionKey.private(message.user_id)
        elif message.chat_type is ChatKind.GROUP:
            if not self._access_policy.is_chat_allowed(message.chat_id):
                return None
            key = SessionKey.group(message.chat_id)
        else:
            return None
        async with self._sessions.lock_for(key):
            await self._sessions.reset(key)
            if self._workspace_service is not None:
                self._workspace_service.reset(message)
            if self._attachment_service is not None:
                attachment_key = self._attachment_key(message)
                if attachment_key is not None:
                    await self._attachment_service.reset(attachment_key)
        return self._phrase_provider.new_session_phrase()

    async def _handle_private(self, message: IncomingMessage) -> str | None:
        if not self._access_policy.is_user_allowed(message.user_id):
            logger.info(
                "Acceso rechazado en privado user_id=%s chat_id=%s",
                message.user_id,
                message.chat_id,
            )
            return self._phrase_provider.rejection_phrase()
        return await self._respond_to_text(message, SessionKey.private(message.user_id))

    async def _handle_group(self, message: IncomingMessage) -> str | None:
        if not self._access_policy.is_chat_allowed(message.chat_id):
            return None
        if not (message.mentions_bot or message.replies_to_bot):
            return None
        return await self._respond_to_text(message, SessionKey.group(message.chat_id))

    async def _respond_to_text(
        self, message: IncomingMessage, key: SessionKey
    ) -> str | None:
        if message.text is None:
            return self._phrase_provider.only_text_phrase()
        prompt = message.text.strip()
        if not prompt:
            # Solo llegó la mención: se invita a preguntar sin gastar cuota.
            if message.mentions_bot:
                return self._phrase_provider.empty_mention_phrase()
            return None
        attachment_key = self._attachment_key(message)
        if self._attachment_service is not None and attachment_key is not None:
            await self._attachment_service.expire_if_stale(attachment_key)
            if self._attachment_service.is_awaiting(attachment_key):
                decision = await self._attachment_service.resolve_intent(attachment_key, prompt)
                if decision.action is not NextAction.PROCESS:
                    return "\n\n".join(decision.notices) if decision.notices else None
                prompt = decision.request or prompt
        async with self._sessions.lock_for(key):
            if self._workspace_service is not None:
                local = await self._workspace_service.confirm(message)
                if local is not None:
                    return local
                local = self._workspace_service.attachment_clarification(message)
                if local is not None:
                    return local
            # Primero expirar la sesión; un resumen de una sesión expirada no debe
            # sobrevivir ni provocar una llamada adicional a la IA.
            window = await self._sessions.load_window(key)
            try:
                preparation = self._skill_service.prepare(prompt)
            except SkillLimitError as exc:
                logger.info(
                    "Límite de skills user_id=%s chat_id=%s motivo=%s",
                    message.user_id,
                    message.chat_id,
                    exc.reason,
                )
                return self._phrase_provider.skill_limit_phrase()
            attempted = await self._sessions.compaction_due(key)
            compacted = (
                await self._sessions.compact_if_due(key, self._ai_client)
                if attempted else False
            )
            if compacted:
                window = await self._sessions.load_window(key)
            turn = self._workspace_service.workspace_for(message) if self._workspace_service else None
            attachments: Sequence[AttachmentPayload] = ()
            if self._attachment_service is not None and attachment_key is not None:
                attachments = await self._attachment_service.payloads(attachment_key)
            contains_audio = any(
                payload.mime_type.split(";", 1)[0].strip().lower().startswith("audio/")
                for payload in attachments
            )
            attachments_consumed = False
            if contains_audio:
                try:
                    attachments = await self._transcribe_audio_attachments(attachments)
                except AudioTranscriptionError as exc:
                    logger.warning(
                        "No se pudo transcribir adjunto user_id=%s chat_id=%s reason=%s",
                        message.user_id, message.chat_id, exc.reason,
                    )
                    if self._attachment_service is not None and attachment_key is not None:
                        await self._attachment_service.reset(attachment_key)
                    return self._audio_error_reply(exc)
            # El historial compartido puede contener instrucciones de miembros sin
            # permiso; tampoco se incorpora texto citado en turnos con tools grupales.
            group_workspace = turn is not None and key.kind is ChatKind.GROUP
            current = with_author(
                key.kind,
                ChatMessage(
                    role=MessageRole.USER,
                    content=preparation.prompt if group_workspace else self._build_prompt(message, preparation.prompt),
                    author_name=message.user_display_name,
                ),
            )
            try:
                if sum(payload_size(item) for item in attachments) > self._attachment_budget_bytes:
                    raise AIError(
                        "presupuesto de adjuntos excedido",
                        user_message=(
                            "Los adjuntos no caben juntos en una sola consulta. "
                            "Usa /reset y vuelve a enviarlos en grupos más pequeños."
                        ),
                    )
                if turn is None:
                    reply = await self._ai_client.complete(
                        [*window, current],
                        skills=preparation.context,
                        attachments=attachments,
                    )
                else:
                    reply = await self._ai_client.complete(
                        [current] if group_workspace else [*window, current],
                        skills=preparation.context,
                        workspace_tools=turn,
                        attachments=attachments,
                    )
            except AIError as exc:
                logger.error(
                    "Fallo al consultar la IA user_id=%s chat_id=%s motivo=%s",
                    message.user_id,
                    message.chat_id,
                    exc.reason,
                )
                if (contains_audio and self._attachment_service is not None
                        and attachment_key is not None):
                    await self._attachment_service.reset(attachment_key)
                    attachments_consumed = True
                error_detail = (exc.user_message or "").strip() or exc.reason.strip()
                if not error_detail:
                    error_detail = "No hay más detalles disponibles."
                error_reply = f"OOps: {error_detail}"
                if turn is not None and turn.notices:
                    error_reply += "\n\n" + "\n\n".join(turn.notices)
                return self._with_notice(error_reply) if compacted else error_reply
            if (contains_audio and self._attachment_service is not None
                    and attachment_key is not None):
                await self._attachment_service.reset(attachment_key)
                attachments_consumed = True
            if (turn is not None and turn.long_read and key.kind is ChatKind.GROUP
                    and any(notice.startswith("El documento ") or "acota la consulta" in notice
                            for notice in turn.notices)):
                reply = (f"El documento {turn.read_ref} es largo para publicarlo en el grupo; "
                         "acota la consulta a un fragmento.")
            if (turn is not None and turn.read_ref is not None and len(reply) > 1900
                    and key.kind is ChatKind.PRIVATE and not turn.long_read):
                turn.notices.append(
                    f"Si prefieres recibir {turn.read_ref} como archivo, pide: envíame {turn.read_ref}."
                )
            # Los resultados y códigos efímeros de tools no entran en la memoria.
            stored_reply = "[Operaciones documentales realizadas; detalles efímeros.]" if turn is not None and turn.notices else reply
            await self._sessions.record(
                key,
                ChatMessage(
                    role=MessageRole.USER,
                    content=preparation.prompt,
                    author_name=message.user_display_name,
                ),
                ChatMessage(role=MessageRole.ASSISTANT, content=stored_reply),
            )
            if (attachments and self._attachment_service is not None
                    and attachment_key is not None and not attachments_consumed):
                await self._attachment_service.reset(attachment_key)
            # Una compactación pendiente que falló al inicio no se reintenta dos
            # veces en el mismo turno. El historial permanece intacto.
            if not attempted:
                compacted = await self._sessions.compact_if_due(key, self._ai_client)
            if turn is not None and turn.notices:
                notices = [notice for notice in turn.notices if notice != reply]
                if notices:
                    reply += "\n\n" + "\n\n".join(notices)
            return self._with_notice(reply) if compacted else reply

    @staticmethod
    def _with_notice(reply: str) -> str:
        return f"{reply}\n\n{COMPACTION_NOTICE}"

    async def _transcribe_audio_attachments(
        self, attachments: Sequence[AttachmentPayload],
    ) -> tuple[AttachmentPayload, ...]:
        """Sustituye binarios de audio por transcript; el AIClient recibe solo texto."""
        if self._audio_transcriber is None:
            raise AudioTranscriptionError("not_configured")
        prepared: list[AttachmentPayload] = []
        for payload in attachments:
            mime_type = payload.mime_type.split(";", 1)[0].strip().lower()
            if not mime_type.startswith("audio/"):
                prepared.append(payload)
                continue
            if payload.mode is not AttachmentMode.BINARY or payload.data is None:
                raise AudioTranscriptionError("invalid_audio_payload")
            transcript = await self._audio_transcriber.transcribe(
                payload.data, mime_type=mime_type,
            )
            if not transcript.strip():
                raise AudioTranscriptionError("empty_transcript")
            prepared.append(AttachmentPayload(
                name=payload.name,
                mime_type="text/plain",
                mode=AttachmentMode.TEXT,
                text=(
                    "Transcripción de voz del archivo. No se proporcionó el audio "
                    "original al modelo; solo se dispone de este texto. Responde "
                    "según lo transcrito. Si la "
                    "petición requiere analizar música, tono, efectos o sonidos no "
                    "verbales, explica que no puedes analizarlos.\n\n"
                    f"{transcript.strip()}"
                ),
            ))
        return tuple(prepared)

    @staticmethod
    def _audio_not_configured_reply() -> str:
        return "El procesamiento de audio no está configurado en este bot."

    @classmethod
    def _audio_error_reply(cls, error: AudioTranscriptionError) -> str:
        if error.reason == "not_configured":
            return cls._audio_not_configured_reply()
        if error.reason in {"empty_transcript", "empty_audio"}:
            return "No detecté voz hablada que pueda transcribir en ese audio."
        if error.reason == "unsupported_format":
            return "No puedo procesar ese formato de audio."
        return "No pude transcribir el audio; vuelve a intentarlo con otro archivo."

    @staticmethod
    def _build_prompt(message: IncomingMessage, prompt: str) -> str:
        """Añade como contexto el mensaje del bot al que se responde, si lo hay."""
        if not message.replies_to_bot or not message.quoted_text:
            return prompt
        quoted = message.quoted_text.strip()
        if not quoted:
            return prompt
        return (
            "Contexto: el asistente respondió antes lo siguiente:\n"
            f"«{quoted}»\n\n"
            f"Pregunta del usuario: {prompt}"
        )
