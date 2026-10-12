"""Tests del reparto de adjuntos por presupuesto de petición (Wave 5, tarea 5.3)."""

from __future__ import annotations

from oraculo.application.budget import split_by_budget
from oraculo.domain.attachments import AttachmentMode, AttachmentPayload


def binary(name: str, size: int) -> AttachmentPayload:
    return AttachmentPayload(
        name=name, mime_type="image/jpeg", mode=AttachmentMode.BINARY,
        data=b"x" * size,
    )


def text(name: str, text_value: str) -> AttachmentPayload:
    return AttachmentPayload(
        name=name, mime_type="text/plain", mode=AttachmentMode.TEXT, text=text_value,
    )


def test_todo_cabe_en_una_sola_tanda():
    payloads = (binary("a.jpg", 10), binary("b.jpg", 20))

    tandas = split_by_budget(payloads, max_bytes=100)

    assert len(tandas) == 1
    assert [p.name for p in tandas[0]] == ["a.jpg", "b.jpg"]


def test_se_divide_en_tandas_sin_superar_el_presupuesto():
    payloads = (binary("a.jpg", 60), binary("b.jpg", 60), binary("c.jpg", 10))

    tandas = split_by_budget(payloads, max_bytes=100)

    assert [[p.name for p in tanda] for tanda in tandas] == [
        ["a.jpg"], ["b.jpg", "c.jpg"],
    ]


def test_el_orden_global_se_conserva():
    payloads = tuple(binary(f"{i}.jpg", 40) for i in range(4))

    tandas = split_by_budget(payloads, max_bytes=100)

    flat = [p.name for tanda in tandas for p in tanda]
    assert flat == ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]


def test_el_texto_cuenta_por_longitud_de_codificacion():
    payloads = (text("a.txt", "x" * 60), text("b.txt", "y" * 60))

    tandas = split_by_budget(payloads, max_bytes=100)

    assert [[p.name for p in tanda] for tanda in tandas] == [["a.txt"], ["b.txt"]]


def test_archivo_individual_mas_grande_que_el_presupuesto_va_solo():
    payloads = (binary("enorme.jpg", 500), binary("pequeña.jpg", 5))

    tandas = split_by_budget(payloads, max_bytes=100)

    assert [[p.name for p in tanda] for tanda in tandas] == [
        ["enorme.jpg"], ["pequeña.jpg"],
    ]


def test_sin_adjuntos_hay_cero_tandas():
    assert split_by_budget((), max_bytes=100) == []


def test_varias_tandas_se_concatenan_con_aviso():
    from oraculo.application.budget import merge_replies

    merged = merge_replies(["respuesta uno", "respuesta dos"])

    assert "respuesta uno" in merged
    assert "respuesta dos" in merged
    assert "tanda" in merged or "archivos" in merged
    assert merged.index("respuesta uno") < merged.index("respuesta dos")


def test_una_sola_respuesta_no_lleva_aviso_de_tandas():
    from oraculo.application.budget import merge_replies

    merged = merge_replies(["única"])

    assert merged == "única"


def test_respuestas_vacias_no_revientan():
    from oraculo.application.budget import merge_replies

    assert merge_replies([]) == ""
