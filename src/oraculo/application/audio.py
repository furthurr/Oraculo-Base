"""Validación y transcripción compartida de notas de voz."""

from __future__ import annotations

from oraculo.application.attachments import MAX_FILE_BYTES, detect_audio_format
from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.errors import AudioTranscriptionError
from oraculo.domain.ports import AudioTranscriber


async def transcribe_voice(
    attachment: IncomingAttachment,
    data: bytes | None,
    *,
    transcriber: AudioTranscriber | None,
) -> str:
    """Valida y transcribe una nota `voice`, devolviendo texto no vacío."""
    if attachment.kind != "voice":
        raise AudioTranscriptionError("unsupported_format")
    if (
        (attachment.file_size is not None and attachment.file_size > MAX_FILE_BYTES)
        or (data is not None and len(data) > MAX_FILE_BYTES)
    ):
        raise AudioTranscriptionError("file_too_large")
    if data is None:
        raise AudioTranscriptionError("download_failed")

    audio_format = detect_audio_format(
        data,
        mime_type=attachment.mime_type or "",
        kind=attachment.kind,
    )
    if audio_format is None:
        raise AudioTranscriptionError("unsupported_format")
    if transcriber is None:
        raise AudioTranscriptionError("not_configured")

    mime_type, _ = audio_format
    transcript = await transcriber.transcribe(data, mime_type=mime_type)
    if not transcript.strip():
        raise AudioTranscriptionError("empty_transcript")
    return transcript.strip()


def format_voice_prompt(caption: str | None, transcript: str) -> str:
    """Combina el caption con la transcripción sin simular acceso al audio."""
    parts: list[str] = []
    if caption and caption.strip():
        parts.append(f"Instrucción del caption:\n{caption.strip()}")
    parts.append(
        "Transcripción de la nota de voz (no se proporcionó el audio original "
        "al modelo):\n"
        f"{transcript.strip()}\n\n"
        "Responde según las palabras transcritas. Si la solicitud requiere "
        "analizar música, tono, efectos o sonidos no verbales, explica que no "
        "puedes analizarlos."
    )
    return "\n\n".join(parts)
