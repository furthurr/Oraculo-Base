"""Tests de los contratos de dominio para skills (Wave 1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from oraculo.domain.errors import SkillLimitError, SkillLoadError
from oraculo.domain.models import (
    AIAssistantMessage,
    AIToolCall,
    AIToolResult,
    Skill,
    SkillCatalogSnapshot,
    SkillTurnContext,
)


def make_skill(name: str) -> Skill:
    return Skill(
        name=name,
        description=f"Descripción de {name}",
        instructions=f"Instrucciones de {name}",
        content_hash=f"hash-{name}",
    )


def test_snapshot_ordena_y_busca_skills_sin_exponer_un_mapping_mutable():
    alpha = make_skill("alpha")
    zeta = make_skill("zeta")

    snapshot = SkillCatalogSnapshot((zeta, alpha))

    assert snapshot.skills == (alpha, zeta)
    assert snapshot.find("alpha") is alpha
    assert snapshot.find("ausente") is None


def test_snapshot_rechaza_nombres_duplicados():
    with pytest.raises(SkillLoadError):
        SkillCatalogSnapshot((make_skill("alpha"), make_skill("alpha")))


def test_modelos_de_skills_son_inmutables():
    skill = make_skill("alpha")

    with pytest.raises(FrozenInstanceError):
        skill.description = "otra"  # type: ignore[misc]


def test_contexto_vacio_tiene_limites_validos_y_no_comparte_colecciones_mutables():
    context = SkillTurnContext.empty()

    assert context.catalog_text == ""
    assert context.explicit_skills == ()
    assert context.loadable_skills == ()
    assert context.autoload_enabled is False
    assert context.max_rounds > 0
    assert context.max_context_chars > 0


@pytest.mark.parametrize(
    "field,value",
    [("max_rounds", 0), ("max_context_chars", 0)],
)
def test_contexto_rechaza_limites_no_positivos(field: str, value: int):
    values: dict[str, int] = {"max_rounds": 1, "max_context_chars": 1}
    values[field] = value

    with pytest.raises(SkillLimitError):
        SkillTurnContext(
            max_rounds=values["max_rounds"],
            max_context_chars=values["max_context_chars"],
        )


def test_dtos_preservan_tool_calls_ids_y_reasoning_content():
    call = AIToolCall(
        tool_call_id="call-1",
        name="load_skill",
        arguments_json='{"name":"alpha"}',
    )
    assistant = AIAssistantMessage(
        content=None,
        tool_calls=(call,),
        reasoning_content="Necesito instrucciones",
    )
    result = AIToolResult(tool_call_id="call-1", content="Instrucciones")

    assert assistant.tool_calls == (call,)
    assert assistant.reasoning_content == "Necesito instrucciones"
    assert result.tool_call_id == call.tool_call_id
