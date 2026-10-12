"""Almacén de proyectos de Pi sobre el sistema de archivos.

Implementa `PiProjectStorePort`. Cada proyecto es una subcarpeta de primer nivel
de la raíz (`PI_WORKSPACE_DIR`). Todo acceso a disco del flujo de selección vive
aquí y está confinado a la raíz: los nombres se validan con un patrón de un solo
segmento y la ruta final se comprueba con `resolve()` + `is_relative_to`, lo que
cubre `..`, rutas absolutas y enlaces simbólicos que escapan.

El confinamiento es lógico (validación en Oraculo), no un sandbox del SO. Los
mensajes al usuario nunca incluyen rutas absolutas.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from oraculo.domain.pi import PiModeError, PiProject

logger = logging.getLogger(__name__)

MAX_PROJECT_NAME_CHARS = 64
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z", re.ASCII)


class FilesystemProjectStore:
    """Lista, resuelve y crea proyectos confinados a `root`."""

    def __init__(self, *, root: Path) -> None:
        self._root = root

    def list_projects(self) -> list[PiProject]:
        root = self._resolved_root()
        try:
            entries = sorted(root.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            logger.warning("Pi: no se pudo listar proyectos (%s)", type(exc).__name__)
            raise PiModeError(
                "projects_unavailable",
                user_message="No se pudo leer la carpeta de proyectos.",
            ) from None
        projects: list[PiProject] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            project = self._confined_dir(root, entry)
            if project is not None:
                projects.append(project)
        return projects

    def resolve(self, name: str) -> PiProject:
        root = self._resolved_root()
        self._validate_name(name)
        project = self._confined_dir(root, root / name)
        if project is None:
            raise PiModeError(
                "project_invalid",
                user_message=f"El proyecto «{name}» no está disponible.",
            )
        return project

    def create(self, name: str) -> tuple[PiProject, bool]:
        root = self._resolved_root()
        self._validate_name(name)
        candidate = root / name
        if candidate.exists() or candidate.is_symlink():
            project = self._confined_dir(root, candidate)
            if project is None:
                raise PiModeError(
                    "project_invalid",
                    user_message=f"Ya existe una entrada «{name}» que no es un proyecto válido.",
                )
            return project, True
        try:
            candidate.mkdir()
        except FileExistsError:
            # Carrera: otro usuario lo creó entre la comprobación y el mkdir.
            return self.resolve(name), True
        except OSError as exc:
            logger.warning("Pi: no se pudo crear el proyecto (%s)", type(exc).__name__)
            raise PiModeError(
                "project_create_failed",
                user_message="No se pudo crear la carpeta del proyecto.",
            ) from None
        return self.resolve(name), False

    def _resolved_root(self) -> Path:
        try:
            return self._root.resolve(strict=True)
        except OSError as exc:
            logger.warning("Pi: raíz de proyectos no disponible (%s)", type(exc).__name__)
            raise PiModeError(
                "projects_unavailable",
                user_message="No se pudo leer la carpeta de proyectos.",
            ) from None

    @staticmethod
    def _validate_name(name: str) -> None:
        if len(name) > MAX_PROJECT_NAME_CHARS or not _PROJECT_NAME.match(name):
            raise PiModeError(
                "project_invalid",
                user_message=(
                    "Nombre de proyecto no válido. Usa letras, números, «-», «_» o «.», "
                    f"empezando por letra o número (máx. {MAX_PROJECT_NAME_CHARS})."
                ),
            )

    @staticmethod
    def _confined_dir(root: Path, entry: Path) -> PiProject | None:
        """Devuelve el proyecto si `entry` es un directorio que resuelve dentro de `root`."""
        try:
            resolved = entry.resolve(strict=True)
        except OSError:
            return None
        if resolved == root or not resolved.is_relative_to(root) or not resolved.is_dir():
            return None
        return PiProject(name=entry.name, path=str(resolved))
