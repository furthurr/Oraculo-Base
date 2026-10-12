"""Tests del catálogo local de skills."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from oraculo.infrastructure.skill_catalog import FileSkillCatalog


SkillFileFactory = Callable[..., Path]


def test_carga_solo_skills_directas_ordenadas(skill_file_factory: SkillFileFactory, tmp_path: Path):
    skill_file_factory(name="zeta", instructions="Instrucciones zeta")
    skill_file_factory(name="alpha", instructions="Instrucciones alpha")
    nested = tmp_path / "grupo" / "nested"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested\ndescription: No debe cargarse.\n---\nContenido\n",
        encoding="utf-8",
    )
    (tmp_path / "SKILL.md").write_text("ignorado", encoding="utf-8")

    catalog = FileSkillCatalog(tmp_path, max_chars=100)

    assert tuple(skill.name for skill in catalog.snapshot.skills) == ("alpha", "zeta")
    assert catalog.snapshot.find("alpha").instructions == "Instrucciones alpha"  # type: ignore[union-attr]
    assert len(catalog.snapshot.find("alpha").content_hash) == 64  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "content,directory",
    [
        ("sin frontmatter", "sin-frontmatter"),
        ("---\n- alpha\n---\nCuerpo", "alpha"),
        ("---\nname: alpha\ndescription: válida\nunknown: value\n---\nCuerpo", "alpha"),
        ("---\nname: alpha\ndescription: válida\nlicense: []\n---\nCuerpo", "alpha"),
        ("---\nname: alpha\ndescription: válida\nmetadata: []\n---\nCuerpo", "alpha"),
        ("---\nname: alpha\ndescription: |\n  dos\n  líneas\n---\nCuerpo", "alpha"),
        ("---\nname: alpha\ndescription: válida\n---\n   \n", "alpha"),
        ("---\nname: otro\ndescription: válida\n---\nCuerpo", "alpha"),
        ("---\nname: Alpha\ndescription: válida\n---\nCuerpo", "Alpha"),
        ("---\nname: !!python/object:builtins.str {}\ndescription: válida\n---\nCuerpo", "alpha"),
    ],
)
def test_omite_frontmatter_o_cuerpo_invalido(
    content: str,
    directory: str,
    tmp_path: Path,
):
    skill_dir = tmp_path / directory
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    catalog = FileSkillCatalog(tmp_path, max_chars=100)

    assert catalog.snapshot.skills == ()


def test_acepta_solo_claves_opcionales_con_tipos_estrictos(tmp_path: Path):
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: alpha\n"
        "description: Skill válida.\n"
        "license: MIT\n"
        "compatibility: Oraculo\n"
        "metadata:\n  owner: operator\n"
        "---\n"
        "Cuerpo\n",
        encoding="utf-8",
    )

    skill = FileSkillCatalog(tmp_path, max_chars=100).snapshot.find("alpha")

    assert skill is not None
    assert skill.license == "MIT"
    assert skill.compatibility == "Oraculo"
    assert skill.metadata == (("owner", "operator"),)


def test_aplica_fronteras_de_descripcion_y_cuerpo(skill_file_factory: SkillFileFactory, tmp_path: Path):
    skill_file_factory(name="exacta", description="d" * 500, instructions="i" * 10)
    skill_file_factory(name="larga", description="d" * 501, instructions="válida")
    skill_file_factory(name="cuerpo-largo", instructions="i" * 11)

    catalog = FileSkillCatalog(tmp_path, max_chars=10)

    assert tuple(skill.name for skill in catalog.snapshot.skills) == ("exacta",)


def test_omite_archivo_que_no_es_utf8(tmp_path: Path):
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_bytes(b"---\nname: alpha\ndescription: ok\n---\n\xff")

    assert FileSkillCatalog(tmp_path, max_chars=100).snapshot.skills == ()


@pytest.mark.parametrize("link_directory", [True, False])
def test_rechaza_symlink_que_resuelve_fuera_de_la_raiz(
    link_directory: bool,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    external = tmp_path_factory.mktemp("external")
    external_file = external / "SKILL.md"
    external_file.write_text(
        "---\nname: alpha\ndescription: externa\n---\nNo leer\n",
        encoding="utf-8",
    )
    if link_directory:
        (tmp_path / "alpha").symlink_to(external, target_is_directory=True)
    else:
        skill_dir = tmp_path / "alpha"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").symlink_to(external_file)

    assert FileSkillCatalog(tmp_path, max_chars=100).snapshot.skills == ()


def test_omite_todos_los_candidatos_de_una_colision(
    skill_file_factory: SkillFileFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    skill_file = skill_file_factory(name="alpha")
    skill_dir = skill_file.parent
    original_iterdir = Path.iterdir

    def duplicate_candidate(path: Path) -> Iterator[Path]:
        if path == tmp_path.resolve():
            return iter((skill_dir, skill_dir))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", duplicate_candidate)

    assert FileSkillCatalog(tmp_path, max_chars=100).snapshot.skills == ()


def test_snapshot_no_depende_de_io_posterior(skill_file_factory: SkillFileFactory, tmp_path: Path):
    skill_file = skill_file_factory(name="alpha", instructions="original")
    catalog = FileSkillCatalog(tmp_path, max_chars=100)
    snapshot = catalog.snapshot

    skill_file.write_text("contenido reemplazado", encoding="utf-8")
    skill_file.unlink()

    assert catalog.snapshot is snapshot
    assert snapshot.find("alpha").instructions == "original"  # type: ignore[union-attr]


def test_directorio_ausente_da_snapshot_vacio(tmp_path: Path):
    catalog = FileSkillCatalog(tmp_path / "ausente", max_chars=100)

    assert catalog.snapshot.skills == ()


def test_logs_solo_contienen_causa_y_conteos_sin_cuerpo_ni_path_absoluto(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    secret_body = "CUERPO_SUPER_SECRETO"
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: alpha\ndescription: válida\nunknown: x\n---\n{secret_body}\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO):
        FileSkillCatalog(tmp_path, max_chars=100)

    logs = caplog.text
    assert "loaded=0" in logs
    assert "skipped=1" in logs
    assert secret_body not in logs
    assert str(tmp_path) not in logs
