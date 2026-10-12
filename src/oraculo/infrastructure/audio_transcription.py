"""Adaptador local de transcripción de audio con faster-whisper."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import TYPE_CHECKING

from oraculo.domain.errors import AudioTranscriptionError
from oraculo.domain.ports import AudioTranscriber

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

DEFAULT_MODEL_SIZE = "base"
SUPPORTED_AUDIO_MIME_TYPES = frozenset({
    "audio/ogg",
    "audio/flac",
    "audio/wav",
    "audio/mpeg",
    "audio/aac",
    "audio/mp4",
    "audio/webm",
})

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(AudioTranscriber):
    """Transcribe audio localmente; carga el modelo solo al primer uso."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._model_lock = threading.Lock()

    async def transcribe(self, data: bytes, *, mime_type: str) -> str:
        """Devuelve texto de voz sin enviar audio a servicios remotos."""
        if not data:
            raise AudioTranscriptionError("empty_audio")

        normalized_mime = mime_type.split(";", 1)[0].strip().lower()
        if normalized_mime not in SUPPORTED_AUDIO_MIME_TYPES:
            raise AudioTranscriptionError("unsupported_format")

        try:
            transcript = await asyncio.to_thread(self._transcribe_sync, data)
        except AudioTranscriptionError:
            raise
        except Exception as exc:
            logger.warning(
                "Falló la transcripción local error_type=%s",
                type(exc).__name__,
            )
            raise AudioTranscriptionError("transcription_failed") from None

        if not transcript:
            raise AudioTranscriptionError("empty_transcript")
        return transcript

    def _transcribe_sync(self, data: bytes) -> str:
        """Carga una vez y consume los segmentos fuera del event loop."""
        with self._model_lock:
            model = self._load_model()
            # segments, _ = model.transcribe(io.BytesIO(data))
            segments, _ = model.transcribe(io.BytesIO(data), language="es")
            return "".join(segment.text for segment in segments).strip()

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                DEFAULT_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
        return self._model
