"""Tests de `FilesystemProjectStore`: listado, resolución, creación y confinamiento."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oraculo.domain.pi import PiModeError, PiProject
from oraculo.infrastructure.pi_projects import FilesystemProjectStore


@pytest.fixture
def root(tmp_path: Path) -> Path:
    base = tmp_path / "pi"
    base.mkdir()
    return base


def _names(projects: list[PiProject]) -> list[str]:
    return [p.name for p in projects]


class TestListProjects:
    def test_empty_root_returns_empty_list(self, root: Path) -> None:
        assert FilesystemProjectStore(root=root).list_projects() == []

    def test_lists_only_first_level_dirs_sorted(self, root: Path) -> None:
        (root / "zeta").mkdir()
        (root / "alfa").mkdir()
        (root / "alfa" / "anidado").mkdir()
        (root / "notas.txt").write_text("x")
        assert _names(FilesystemProjectStore(root=root).list_projects()) == ["alfa", "zeta"]

    def test_ignores_hidden_entries(self, root: Path) -> None:
        (root / ".git").mkdir()
        (root / "visible").mkdir()
        assert _names(FilesystemProjectStore(root=root).list_projects()) == ["visible"]

    def test_paths_are_absolute_and_inside_root(self, root: Path) -> None:
        (root / "demo").mkdir()
        [project] = FilesystemProjectStore(root=root).list_projects()
        assert Path(project.path).is_absolute()
        assert Path(project.path) == (root / "demo").resolve()

    def test_excludes_symlink_escaping_root(self, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "fuera"
        outside.mkdir()
        os.symlink(outside, root / "escape")
        (root / "ok").mkdir()
        assert _names(FilesystemProjectStore(root=root).list_projects()) == ["ok"]

    def test_keeps_symlink_inside_root(self, root: Path) -> None:
        (root / "real").mkdir()
        os.symlink(root / "real", root / "alias")
        assert _names(FilesystemProjectStore(root=root).list_projects()) == ["alias", "real"]

    def test_missing_root_raises_typed_error(self, tmp_path: Path) -> None:
        store = FilesystemProjectStore(root=tmp_path / "no-existe")
        with pytest.raises(PiModeError) as exc:
            store.list_projects()
        assert exc.value.reason == "projects_unavailable"
        assert exc.value.user_message is not None
        assert str(tmp_path) not in exc.value.user_message


INVALID_NAMES = ["", " ", ".", "..", "../fuera", "a/b", "a\\b", "/abs", ".oculto", "con espacio", "ñandú"]


class TestResolve:
    def test_resolves_existing_project(self, root: Path) -> None:
        (root / "demo").mkdir()
        project = FilesystemProjectStore(root=root).resolve("demo")
        assert project == PiProject(name="demo", path=str((root / "demo").resolve()))

    @pytest.mark.parametrize("name", INVALID_NAMES)
    def test_rejects_invalid_names(self, root: Path, name: str) -> None:
        with pytest.raises(PiModeError) as exc:
            FilesystemProjectStore(root=root).resolve(name)
        assert exc.value.reason == "project_invalid"

    def test_rejects_missing_project(self, root: Path) -> None:
        with pytest.raises(PiModeError) as exc:
            FilesystemProjectStore(root=root).resolve("fantasma")
        assert exc.value.reason == "project_invalid"

    def test_rejects_symlink_escaping_root(self, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "fuera"
        outside.mkdir()
        os.symlink(outside, root / "escape")
        with pytest.raises(PiModeError) as exc:
            FilesystemProjectStore(root=root).resolve("escape")
        assert exc.value.reason == "project_invalid"


class TestCreate:
    def test_creates_new_project(self, root: Path) -> None:
        project, existed = FilesystemProjectStore(root=root).create("nuevo")
        assert existed is False
        assert (root / "nuevo").is_dir()
        assert project.path == str((root / "nuevo").resolve())

    def test_reuses_existing_project(self, root: Path) -> None:
        (root / "demo").mkdir()
        (root / "demo" / "archivo.txt").write_text("conservar")
        project, existed = FilesystemProjectStore(root=root).create("demo")
        assert existed is True
        assert (root / "demo" / "archivo.txt").read_text() == "conservar"
        assert project.name == "demo"

    @pytest.mark.parametrize("name", INVALID_NAMES)
    def test_rejects_invalid_names_without_creating(self, root: Path, name: str) -> None:
        with pytest.raises(PiModeError) as exc:
            FilesystemProjectStore(root=root).create(name)
        assert exc.value.reason == "project_invalid"
        assert list(root.iterdir()) == []

    def test_name_colliding_with_file_is_rejected(self, root: Path) -> None:
        (root / "archivo").write_text("x")
        with pytest.raises(PiModeError) as exc:
            FilesystemProjectStore(root=root).create("archivo")
        assert exc.value.reason == "project_invalid"

    def test_os_error_is_translated(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(self: Path, *args: object, **kwargs: object) -> None:
            raise PermissionError("denegado")

        monkeypatch.setattr(Path, "mkdir", boom)
        with pytest.raises(PiModeError) as exc:
            FilesystemProjectStore(root=root).create("nuevo")
        assert exc.value.reason == "project_create_failed"
        assert exc.value.user_message is not None
        assert str(root) not in exc.value.user_message
