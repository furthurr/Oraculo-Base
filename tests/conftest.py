"""Fixtures compartidas para archivos locales de skills."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Desvía la raíz del proyecto de `load_config` hacia un directorio temporal.

    `infrastructure/config.py` calcula la raíz del proyecto desde su propia
    ruta (`Path(__file__).resolve().parents[3]`); redirigir `__file__` hace que
    `<raíz-del-proyecto>/workspace` sea `tmp_path/workspace` sin tocar el
    workspace real del repositorio.
    """
    import oraculo.infrastructure.config as config_module

    fake = tmp_path / "src" / "oraculo" / "infrastructure" / "config.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_module, "__file__", str(fake))
    return tmp_path


@pytest.fixture
def skill_file_factory(tmp_path: Path) -> Callable[..., Path]:
    """Crea un `<raíz>/<slug>/SKILL.md` configurable para tests del loader."""

    def create(
        name: str = "resumir",
        *,
        description: str = "Resume texto.",
        instructions: str = "Resume el contenido de forma breve.",
        directory_name: str | None = None,
        frontmatter_extra: str = "",
    ) -> Path:
        skill_dir = tmp_path / (directory_name or name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        extra = f"{frontmatter_extra.rstrip()}\n" if frontmatter_extra else ""
        skill_file.write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"{extra}"
            "---\n"
            f"{instructions}\n",
            encoding="utf-8",
        )
        return skill_file

    return create
