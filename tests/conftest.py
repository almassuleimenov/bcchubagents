"""Фикстуры: битые и краевые файлы генерируются на лету (не храним бинарники в Git)."""

from __future__ import annotations

import functools
import io
import os
from pathlib import Path

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MD = ROOT / "docs" / "requirements.md"
FONT = next(iter(sorted((ROOT / "assets" / "fonts").glob("*.ttf"))), None)

RU_TEXT = (
    "# Требования\n\n## Функциональные требования\n\n"
    "* **Ф-1.** Система должна быстро сохранять порог.\n"
    "* **Ф-2.** При возможности система отправляет push и т.д.\n"
)


@pytest.fixture(scope="session")
def sample_md() -> Path:
    return SAMPLE_MD


@functools.lru_cache(maxsize=None)
def _pdf_bytes(text: str | None = "Требования: система должна сохранять порог. " * 3, *, encrypt: bool = False,
               image: bool = False) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        if FONT is not None:
            page.insert_font(fontname="ru", fontfile=str(FONT))
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontname="ru", fontsize=11)
        else:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), "Requirements: the system shall save. " * 5, fontsize=11)
    if image:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 400), False)
        pix.set_rect(pix.irect, (230, 230, 230))
        page.insert_image(page.rect, pixmap=pix)
    kw = {}
    if encrypt:
        kw = dict(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    data = doc.tobytes(**kw)
    doc.close()
    return data


@pytest.fixture()
def broken_files(tmp_path: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def put(name: str, data: bytes) -> None:
        p = tmp_path / name
        p.write_bytes(data)
        files[name] = p

    put("empty.md", b"")
    put("png_as.md", b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + os.urandom(256))
    put("cp1251.md", RU_TEXT.encode("cp1251"))
    put("utf16.txt", RU_TEXT.encode("utf-16"))
    good_pdf = _pdf_bytes()
    put("good.pdf", good_pdf)
    put("truncated.pdf", good_pdf[: len(good_pdf) // 3])
    put("encrypted.pdf", _pdf_bytes(encrypt=True))
    put("scanned.pdf", _pdf_bytes(text=None, image=True))
    put("blank.pdf", _pdf_bytes(text=None))
    put("garbage.md", os.urandom(2048))
    put("fake.docx", b"PK\x03\x04" + os.urandom(200))
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "not a docx")
    put("zip_not_docx.docx", buf.getvalue())
    put("whitespace.md", b"   \n\n\t  \n")
    return files


@pytest.fixture()
def docx_file(tmp_path: Path) -> Path:
    import docx

    d = docx.Document()
    d.add_heading("Требования к сервису", level=1)
    d.add_paragraph("Ф-1. Система должна сохранять порог в тиынах.")
    t = d.add_table(rows=2, cols=3)
    t.rows[0].cells[0].text = "Исходные условия"
    t.rows[0].cells[1].text = "Действие"
    t.rows[0].cells[2].text = "Ожидаемый результат"
    t.rows[1].cells[0].text = "Порог 50 000 ₸"
    t.rows[1].cells[1].text = "Покупка 1 500 ₸"
    t.rows[1].cells[2].text = "Один push"
    d.add_paragraph("Ф-2. После таблицы идёт второе требование.")
    p = tmp_path / "req.docx"
    d.save(p)
    return p
