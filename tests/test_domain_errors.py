from __future__ import annotations

from oraculo.domain import errors
from oraculo.domain.errors import AIError


def test_ai_error_separa_el_motivo_interno_del_mensaje_publico() -> None:
    error = AIError("estado 429", user_message="cuota agotada")

    assert error.reason == "estado 429"
    assert error.user_message == "cuota agotada"
    assert str(error) == "Error de IA: estado 429"


def test_ai_error_sin_mensaje_publico_conserva_la_compatibilidad() -> None:
    error = AIError("tiempo de espera agotado")

    assert error.reason == "tiempo de espera agotado"
    assert error.user_message is None


def test_audio_transcription_error_expone_motivo_tipado_sin_detalle_remoto() -> None:
    error_type = getattr(errors, "AudioTranscriptionError", None)
    assert error_type is not None
    error = error_type("timeout")

    assert error.reason == "timeout"
    assert str(error) == "Error de transcripción de audio: timeout"
