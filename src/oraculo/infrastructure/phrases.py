"""Adaptador `PhraseProvider` con selección aleatoria."""

from __future__ import annotations

import random
from collections.abc import Sequence


class RandomPhraseProvider:
    """Devuelve una frase aleatoria de la lista configurada para cada caso."""

    def __init__(
        self,
        *,
        rejection: Sequence[str],
        only_text: Sequence[str],
        empty_mention: Sequence[str],
        new_session: Sequence[str],
        skill_limit: str,
        rng: random.Random | None = None,
    ) -> None:
        self._rejection = self._validated(rejection, "rechazo")
        self._only_text = self._validated(only_text, "mensajes no texto")
        self._empty_mention = self._validated(empty_mention, "solo mención")
        self._new_session = self._validated(new_session, "sesión nueva")
        if not skill_limit.strip():
            raise ValueError("La frase de límite de skills no puede estar vacía")
        self._skill_limit = skill_limit
        self._rng = rng if rng is not None else random.Random()

    @staticmethod
    def _validated(phrases: Sequence[str], name: str) -> tuple[str, ...]:
        if not phrases:
            raise ValueError(f"La lista de frases de {name} no puede estar vacía")
        return tuple(phrases)

    def rejection_phrase(self) -> str:
        return self._rng.choice(self._rejection)

    def only_text_phrase(self) -> str:
        return self._rng.choice(self._only_text)

    def empty_mention_phrase(self) -> str:
        return self._rng.choice(self._empty_mention)

    def new_session_phrase(self) -> str:
        return self._rng.choice(self._new_session)

    def skill_limit_phrase(self) -> str:
        return self._skill_limit
