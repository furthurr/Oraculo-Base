"""Errores tipados del dominio.

Ningún error incluye valores de secretos (tokens o claves de API).
"""

from __future__ import annotations


class ConfigError(Exception):
    """Variable de entorno obligatoria ausente o con formato inválido."""

    def __init__(self, variable: str, reason: str = "no está definida") -> None:
        self.variable = variable
        self.reason = reason
        super().__init__(f"Configuración inválida: {variable} ({reason})")


class AIError(Exception):
    """Fallo tipado de IA con motivo interno y mensaje público opcional."""

    def __init__(self, reason: str, *, user_message: str | None = None) -> None:
        self.reason = reason
        self.user_message = user_message
        super().__init__(f"Error de IA: {reason}")


class SkillLoadError(Exception):
    """Una skill o un snapshot no cumple el contrato de carga."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Error al cargar skill: {reason}")


class SkillLimitError(Exception):
    """Una operación con skills excede un límite seguro."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Límite de skills excedido: {reason}")


class AttachmentError(Exception):
    """Un archivo adjunto incumple un contrato o un límite seguro."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Error de adjunto: {reason}")


class AudioTranscriptionError(Exception):
    """Fallo tipado al preparar o transcribir contenido de audio."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Error de transcripción de audio: {reason}")
