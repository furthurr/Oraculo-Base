from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.errors import AudioTranscriptionError
from oraculo.application.audio import transcribe_voice


@dataclass
class FakeTranscriber:
    transcript: str = "  hola desde voz  "
    error: AudioTranscriptionError | None = None
    calls: list[tuple[bytes, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.calls = []

    async def transcribe(self, data: bytes, *, mime_type: str) -> str:
        self.calls.append((data, mime_type))
        if self.error is not None:
            raise self.error
        return self.transcript


def voice(*, file_size: int | None = None, mime_type: str | None = None) -> IncomingAttachment:
    return IncomingAttachment(
        file_id="voice-file-id",
        file_name=None,
        mime_type=mime_type,
        file_size=file_size,
        kind="voice",
    )


async def test_transcribe_voice_detects_format_and_returns_trimmed_text() -> None:
    transcriber = FakeTranscriber()
    audio = b"OggSvoice"

    transcript = await transcribe_voice(
        voice(file_size=len(audio)), audio, transcriber=transcriber,
    )

    assert transcript == "hola desde voz"
    assert transcriber.calls == [(audio, "audio/ogg")]


@pytest.mark.parametrize(
    ("attachment", "data"),
    [
        (voice(file_size=20 * 1024 * 1024 + 1), b"OggSvoice"),
        (voice(), b"OggS" + b"x" * (20 * 1024 * 1024)),
    ],
)
async def test_transcribe_voice_rejects_declared_or_actual_size_over_limit(
    attachment: IncomingAttachment, data: bytes,
) -> None:
    transcriber = FakeTranscriber()

    with pytest.raises(AudioTranscriptionError, match="file_too_large"):
        await transcribe_voice(attachment, data, transcriber=transcriber)

    assert transcriber.calls == []


async def test_transcribe_voice_rejects_failed_download_without_calling_transcriber() -> None:
    transcriber = FakeTranscriber()

    with pytest.raises(AudioTranscriptionError, match="download_failed"):
        await transcribe_voice(voice(), None, transcriber=transcriber)

    assert transcriber.calls == []


async def test_transcribe_voice_rejects_unknown_format_without_calling_transcriber() -> None:
    transcriber = FakeTranscriber()

    with pytest.raises(AudioTranscriptionError, match="unsupported_format"):
        await transcribe_voice(voice(), b"not an audio file", transcriber=transcriber)

    assert transcriber.calls == []


async def test_transcribe_voice_rejects_empty_transcript() -> None:
    transcriber = FakeTranscriber(transcript="  ")

    with pytest.raises(AudioTranscriptionError, match="empty_transcript"):
        await transcribe_voice(voice(), b"OggSvoice", transcriber=transcriber)

    assert transcriber.calls == [(b"OggSvoice", "audio/ogg")]


async def test_transcribe_voice_preserves_typed_transcriber_error() -> None:
    error = AudioTranscriptionError("transcription_failed")
    transcriber = FakeTranscriber(error=error)

    with pytest.raises(AudioTranscriptionError) as exc_info:
        await transcribe_voice(voice(), b"OggSvoice", transcriber=transcriber)

    assert exc_info.value is error


async def test_transcribe_voice_rejects_non_voice_attachments() -> None:
    transcriber = FakeTranscriber()
    attachment = IncomingAttachment(
        file_id="audio-file-id",
        file_name="recording.ogg",
        mime_type="audio/ogg",
        file_size=8,
        kind="audio",
    )

    with pytest.raises(AudioTranscriptionError, match="unsupported_format"):
        await transcribe_voice(attachment, b"OggSdata", transcriber=transcriber)

    assert transcriber.calls == []
