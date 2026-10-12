"""Almacenamiento temporal de adjuntos por sesión en disco.

Solo este módulo toca el filesystem de adjuntos. Los nombres se sanean para
impedir path traversal y el I/O pesado se ejecuta fuera del event loop.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path, PurePosixPath

from oraculo.domain.attachments import AttachmentMode
from oraculo.domain.errors import AttachmentError
from oraculo.domain.models import AttachmentKey

_UNSAFE = re.compile(r"[^\w.\- ]", re.UNICODE)


def sanitize_name(name: str) -> str:
    """Devuelve un nombre de archivo seguro o lanza `AttachmentError`."""
    if not isinstance(name, str) or not name:
        raise AttachmentError("invalid_name")
    # Solo el componente final; elimina separadores y puntos suspensivos.
    base = PurePosixPath(name.replace("\\", "/")).name
    base = base.strip()
    if not base or base in {".", ".."} or "/" in base or "\\" in base:
        raise AttachmentError("invalid_name")
    if base.startswith("."):
        base = base.lstrip(".") or "archivo"
    # Sustituye caracteres no seguros por guion bajo.
    cleaned = _UNSAFE.sub("_", base)[:120]
    if not cleaned or cleaned in {".", ".."}:
        raise AttachmentError("invalid_name")
    return cleaned


class TempAttachmentStore:
    """Implementación en disco del puerto `AttachmentStore`."""

    def __init__(self, *, root: Path) -> None:
        self._root = root

    def _context_dir(self, key: AttachmentKey) -> Path:
        # `key.value` es un identificador chat/usuario que no admite separadores.
        safe = _UNSAFE.sub("_", key.value)
        return self._root / safe

    def _path(self, key: AttachmentKey, name: str, *, strict: bool) -> Path:
        cleaned = sanitize_name(name)
        if strict and cleaned != name:
            raise AttachmentError("invalid_name")
        directory = self._context_dir(key)
        target = (directory / cleaned).resolve()
        if self._root.resolve() not in target.parents:
            raise AttachmentError("invalid_name")
        return target

    def _write(self, key: AttachmentKey, name: str, data: bytes) -> None:
        target = self._path(key, name, strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def _read(self, key: AttachmentKey, name: str) -> bytes:
        target = self._path(key, name, strict=True)
        if not target.is_file():
            raise AttachmentError("not_found")
        return target.read_bytes()

    def _names(self, key: AttachmentKey) -> tuple[str, ...]:
        directory = self._context_dir(key)
        if not directory.is_dir():
            return ()
        return tuple(sorted(entry.name for entry in directory.iterdir() if entry.is_file()))

    def _remove(self, key: AttachmentKey, name: str) -> None:
        try:
            target = self._path(key, name, strict=True)
        except AttachmentError:
            return
        if target.is_file():
            target.unlink()

    def _clear(self, key: AttachmentKey) -> None:
        directory = self._context_dir(key)
        if not directory.is_dir():
            return
        for entry in directory.iterdir():
            if entry.is_file():
                entry.unlink()
        directory.rmdir()

    def _clear_all(self) -> None:
        if not self._root.is_dir():
            return
        for directory in self._root.iterdir():
            if directory.is_dir():
                for entry in directory.iterdir():
                    if entry.is_file():
                        entry.unlink()
                directory.rmdir()

    async def store(
        self,
        key: AttachmentKey,
        name: str,
        mime_type: str,
        mode: AttachmentMode,
        data: bytes,
    ) -> None:
        if not isinstance(data, bytes):
            raise AttachmentError("invalid_attachment")
        await asyncio.to_thread(self._write, key, name, data)

    async def load(self, key: AttachmentKey, name: str) -> bytes:
        return await asyncio.to_thread(self._read, key, name)

    async def list(self, key: AttachmentKey) -> tuple[str, ...]:
        return await asyncio.to_thread(self._names, key)

    async def delete(self, key: AttachmentKey, name: str) -> None:
        await asyncio.to_thread(self._remove, key, name)

    async def clear(self, key: AttachmentKey) -> None:
        await asyncio.to_thread(self._clear, key)

    async def clear_all(self) -> None:
        await asyncio.to_thread(self._clear_all)
