"""Tests de preparación efímera de skills por turno."""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError

import pytest

from oraculo.application.skills import SkillService
from oraculo.domain.errors import SkillLimitError
from oraculo.domain.models import Skill, SkillCatalogSnapshot


def make_skill(name: str, *, description: str | None = None, instructions: str | None = None) -> Skill:
    return Skill(
        name=name,
        description=description or f"Descripción de {name}",
        instructions=instructions or f"Instrucciones de {name}",
        content_hash=f"hash-{name}",
    )


def make_service(
    *skills: Skill,
    autoload_enabled: bool = True,
    catalog_max_chars: int = 1_000,
    max_context_chars: int = 2_000,
    max_per_turn: int = 3,
) -> SkillService:
    return SkillService(
        SkillCatalogSnapshot(tuple(skills)),
        autoload_enabled=autoload_enabled,
        catalog_max_chars=catalog_max_chars,
        max_context_chars=max_context_chars,
        max_per_turn=max_per_turn,
        max_rounds=3,
    )


def test_catalogo_es_ordenado_delimitado_y_no_incluye_cuerpos():
    service = make_service(
        make_skill("zeta", instructions="CUERPO_ZETA"),
        make_skill("alpha", instructions="CUERPO_ALPHA"),
    )

    context = service.prepare("hola").context

    assert context.catalog_text.startswith("<skills_catalog>\n")
    assert context.catalog_text.endswith("\n</skills_catalog>")
    assert context.catalog_text.index("$alpha") < context.catalog_text.index("$zeta")
    assert "Descripción de alpha" in context.catalog_text
    assert "CUERPO_ALPHA" not in context.catalog_text
    assert context.loadable_skills == service.snapshot.skills


def test_catalogo_recorta_descripciones_deterministicamente_bajo_presupuesto():
    skills = tuple(
        make_skill(name, description=f"Descripción extensa para {name} " * 8)
        for name in ("alpha", "beta", "gamma")
    )
    first = make_service(*skills, catalog_max_chars=150).prepare("hola").context.catalog_text
    second = make_service(*reversed(skills), catalog_max_chars=150).prepare("hola").context.catalog_text

    assert first == second
    assert len(first) <= 150
    assert all(f"${name}" in first for name in ("alpha", "beta", "gamma"))
    assert "trunc" in first.lower()


def test_catalogo_indica_ids_omitidos_si_ni_solos_caben():
    skills = tuple(make_skill(f"skill-{index:02d}") for index in range(20))

    catalog = make_service(*skills, catalog_max_chars=100).prepare("hola").context.catalog_text

    assert len(catalog) <= 100
    assert "$skill-00" in catalog
    assert "omit" in catalog.lower()
    assert "$skill-19" not in catalog


def test_prepare_reconoce_tokens_completos_deduplica_y_conserva_prompt_original():
    alpha = make_skill("alpha")
    beta = make_skill("beta")
    prompt = "$beta, $alpha y otra vez $beta; no $alpha-extra ni x$alpha ni $unknown."

    preparation = make_service(alpha, beta).prepare(prompt)

    assert preparation.prompt == prompt
    assert preparation.context.explicit_skills == (beta, alpha)


def test_skill_desconocida_permanece_como_texto_ordinario():
    preparation = make_service(make_skill("alpha")).prepare("usa $unknown")

    assert preparation.prompt == "usa $unknown"
    assert preparation.context.explicit_skills == ()


def test_exceso_de_skills_unicas_lanza_error_pero_duplicados_no_cuentan():
    alpha = make_skill("alpha")
    beta = make_skill("beta")
    service = make_service(alpha, beta, max_per_turn=1)

    assert service.prepare("$alpha $alpha").context.explicit_skills == (alpha,)
    with pytest.raises(SkillLimitError):
        service.prepare("$alpha $beta")


def test_exceso_de_contexto_explicito_lanza_error():
    skill = make_skill("alpha", instructions="x" * 100)
    service = make_service(skill, catalog_max_chars=80, max_context_chars=150)

    with pytest.raises(SkillLimitError):
        service.prepare("$alpha")


def test_autoload_deshabilitado_no_impide_invocacion_explicita():
    skill = make_skill("alpha")

    context = make_service(skill, autoload_enabled=False).prepare("$alpha").context

    assert context.autoload_enabled is False
    assert context.loadable_skills == ()
    assert context.explicit_skills == (skill,)


def test_preparacion_y_contexto_son_inmutables():
    preparation = make_service(make_skill("alpha")).prepare("hola")

    with pytest.raises(FrozenInstanceError):
        preparation.prompt = "otro"  # type: ignore[misc]


def test_log_de_carga_explicita_no_expone_prompt_ni_instrucciones(
    caplog: pytest.LogCaptureFixture,
):
    secret_prompt = "PROMPT_SECRETO"
    secret_body = "CUERPO_SECRETO"
    service = make_service(make_skill("alpha", instructions=secret_body))

    with caplog.at_level(logging.INFO):
        service.prepare(f"{secret_prompt} $alpha")

    assert "skill_name=alpha" in caplog.text
    assert "mode=explicit" in caplog.text
    assert secret_prompt not in caplog.text
    assert secret_body not in caplog.text
