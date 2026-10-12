"""Tests de política de acceso y frases automáticas (Req 4.1 y spec base)."""

from __future__ import annotations

import random

import pytest

from oraculo.infrastructure.access import EnvAccessPolicy
from oraculo.infrastructure.phrases import RandomPhraseProvider


def test_politica_de_acceso_sobre_listas():
    policy = EnvAccessPolicy(frozenset({1, 2}), frozenset({-100}))

    assert policy.is_user_allowed(1) is True
    assert policy.is_user_allowed(3) is False
    assert policy.is_chat_allowed(-100) is True
    assert policy.is_chat_allowed(-200) is False


def _provider(**overrides) -> RandomPhraseProvider:
    lists = {
        "rejection": ("r1", "r2"),
        "only_text": ("t1", "t2"),
        "empty_mention": ("m1", "m2"),
        "new_session": ("n1", "n2"),
        "skill_limit": "límite configurado",
    }
    lists.update(overrides)
    return RandomPhraseProvider(rng=random.Random(7), **lists)


def test_frase_siempre_pertenece_a_la_lista():
    provider = _provider()

    for _ in range(30):
        assert provider.rejection_phrase() in ("r1", "r2")
        assert provider.only_text_phrase() in ("t1", "t2")
        assert provider.empty_mention_phrase() in ("m1", "m2")
        assert provider.new_session_phrase() in ("n1", "n2")
        assert provider.skill_limit_phrase() == "límite configurado"


@pytest.mark.parametrize(
    "name", ["rejection", "only_text", "empty_mention", "new_session"]
)
def test_lista_de_frases_vacia_no_es_valida(name):
    with pytest.raises(ValueError):
        _provider(**{name: ()})


def test_proveedor_de_frases_no_expone_una_frase_aleatoria_de_error_ia():
    assert not hasattr(_provider(), "ai_error_phrase")
