"""Contratos inmutables para documentos confinados por chat."""

from __future__ import annotations

import re
from dataclasses import dataclass

from oraculo.domain.models import ChatKind


class WorkspaceError(Exception):
    """Fallo de workspace con motivo seguro, sin rutas ni contenido."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Error de workspace: {reason}")


@dataclass(frozen=True)
class WorkspaceScope:
    kind: ChatKind
    id: int

    def __post_init__(self) -> None:
        if self.kind not in (ChatKind.PRIVATE, ChatKind.GROUP) or type(self.id) is not int:
            raise WorkspaceError("invalid_scope")
        if self.kind is ChatKind.PRIVATE and self.id <= 0:
            raise WorkspaceError("invalid_scope")
        if self.kind is ChatKind.GROUP and self.id >= 0:
            raise WorkspaceError("invalid_scope")


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:md|txt)\Z", re.ASCII)


@dataclass(frozen=True)
class DocumentRef:
    name: str
    temporary: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not isinstance(self.temporary, bool)
            or len(self.name) > 64
            or ".." in self.name
            or _NAME.fullmatch(self.name) is None
        ):
            raise WorkspaceError("invalid_reference")


@dataclass(frozen=True)
class DocumentVersion:
    """Identidad del inode y digest del contenido leído para CAS lógico."""

    device: int
    inode: int
    sha256: str
