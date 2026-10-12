"""Compatibilidad del decodificador PyAV incluido con faster-whisper."""

from __future__ import annotations

from io import BytesIO
import wave

import av


def make_wav_bytes() -> bytes:
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(bytes(320))
    return stream.getvalue()


def make_ogg_opus_bytes() -> bytes:
    stream_bytes = BytesIO()
    with av.open(stream_bytes, mode="w", format="ogg") as container:
        audio_stream = container.add_stream("libopus", rate=16_000)
        audio_stream.layout = "mono"
        frame = av.AudioFrame(format="s16", layout="mono", samples=1_600)
        frame.sample_rate = 16_000
        frame.planes[0].update(bytes(frame.planes[0].buffer_size))
        for packet in audio_stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode(None):
            container.mux(packet)
    return stream_bytes.getvalue()


def test_pyav_decodifica_wav_desde_memoria_sin_ffmpeg_externo() -> None:
    with av.open(BytesIO(make_wav_bytes()), mode="r") as container:
        frames = list(container.decode(audio=0))

    assert frames
    assert sum(frame.samples for frame in frames) == 160


def test_pyav_decodifica_opus_ogg_desde_memoria() -> None:
    with av.open(BytesIO(make_ogg_opus_bytes()), mode="r") as container:
        frames = list(container.decode(audio=0))

    assert frames
    assert sum(frame.samples for frame in frames) > 0
