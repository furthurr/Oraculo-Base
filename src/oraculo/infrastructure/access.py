"""Adaptador `AccessPolicy` basado en listas de IDs permitidos."""

from __future__ import annotations


class EnvAccessPolicy:
    """Política de acceso construida desde la configuración."""

    def __init__(
        self,
        allowed_user_ids: frozenset[int],
        allowed_chat_ids: frozenset[int],
    ) -> None:
        self._allowed_user_ids = allowed_user_ids
        self._allowed_chat_ids = allowed_chat_ids

    def is_user_allowed(self, user_id: int) -> bool:
        return user_id in self._allowed_user_ids

    def is_chat_allowed(self, chat_id: int) -> bool:
        return chat_id in self._allowed_chat_ids
