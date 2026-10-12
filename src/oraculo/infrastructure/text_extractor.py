"""Extracción local de texto de adjuntos (Opción A).

Formatos de texto (.txt, .md, .csv, .json y código fuente) y PDF. Cualquier
otro tipo se rechaza como `unsupported` para que la capa de aplicación lo
elimine y avise al usuario.
"""

from __future__ import annotations

import io
from pathlib import PurePosixPath

from oraculo.domain.errors import AttachmentError

_TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".html",
    ".htm", ".log", ".rst", ".ini", ".cfg", ".conf", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".kt", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".sh", ".bat", ".sql",
})


def _suffix(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name
    return PurePosixPath(base).suffix.lower()


class TextExtractor:
    """Extrae texto plano de un adjunto, con presupuesto de caracteres."""

    def __init__(self, *, max_chars: int) -> None:
        if max_chars < 1:
            raise AttachmentError("invalid_attachment")
        self._max_chars = max_chars

    def extract(self, name: str, data: bytes) -> str:
        """Devuelve el texto del archivo truncado al presupuesto.

        Lanza `AttachmentError("unsupported")` si el tipo no permite extracción.
        """
        if not isinstance(data, bytes):
            raise AttachmentError("invalid_attachment")
        suffix = _suffix(name)
        if suffix == ".pdf":
            text = self._from_pdf(data)
        elif suffix in _TEXT_SUFFIXES or not suffix:
            text = self._from_text(data)
        else:
            raise AttachmentError("unsupported")
        return text[: self._max_chars]

    @staticmethod
    def _from_text(data: bytes) -> str:
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _from_pdf(data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise AttachmentError("unsupported") from None
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            # Cualquier PDF ileviable se trata como tipo no procesable.
            raise AttachmentError("unsupported") from None
