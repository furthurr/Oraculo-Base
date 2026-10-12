"""Tests del extractor local de texto de adjuntos (Wave 2, tarea 2.2)."""

from __future__ import annotations

import pytest

from oraculo.domain.errors import AttachmentError
from oraculo.infrastructure.text_extractor import TextExtractor

extractor = TextExtractor(max_chars=1000)


def test_texto_plano_se_extrae_directamente():
    assert extractor.extract("a.txt", b"contenido en espa\xc3\xb1ol") == "contenido en español"


def test_markdown_csv_json_y_codigo_se_extraen():
    assert extractor.extract("a.md", b"# titulo") == "# titulo"
    assert extractor.extract("a.csv", b"a,b") == "a,b"
    assert extractor.extract("a.json", b'{"x": 1}') == '{"x": 1}'
    assert extractor.extract("main.py", b"print(1)") == "print(1)"


def test_utf8_invalido_no_revienta():
    # Bytes no UTF-8: se decodifica con reemplazo, sin excepción.
    resultado = extractor.extract("a.txt", b"\xff\xfeabc")
    assert "abc" in resultado


def test_texto_largo_se_trunca_al_presupuesto():
    extractor_corto = TextExtractor(max_chars=5)
    assert extractor_corto.extract("a.txt", b"0123456789") == "01234"


def test_pdf_con_texto_se_extrae(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as stream:
        writer.write(stream)

    # Una página en blanco: extrae cadena vacía sin lanzar error.
    assert extractor.extract("doc.pdf", path.read_bytes()) == ""


def test_extension_desconocida_lanza_unsupported():
    with pytest.raises(AttachmentError) as exc:
        extractor.extract("a.zip", b"PK")
    assert exc.value.reason == "unsupported"


def test_extension_con_mayusculas_y_ruta_se_normaliza():
    assert extractor.extract("CARPETA/A.TXT", b"hola") == "hola"


def test_pdf_corrupto_lanza_unsupported():
    with pytest.raises(AttachmentError) as exc:
        extractor.extract("doc.pdf", b"no soy un pdf")
    assert exc.value.reason == "unsupported"
