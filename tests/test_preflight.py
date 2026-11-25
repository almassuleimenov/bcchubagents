from srs_audit.parsers import parse_document
from srs_audit.preflight import run_preflight


def codes(res):
    return {i.code for i in res.health.issues}


def test_zero_bytes_is_unprocessable(broken_files):
    res = run_preflight(broken_files["empty.md"])
    assert res.health.status == "UNPROCESSABLE"
    assert "EMPTY_FILE" in codes(res)


def test_png_disguised_as_md(broken_files):
    res = run_preflight(broken_files["png_as.md"])
    assert res.health.status == "UNPROCESSABLE"
    assert "IMAGE_NOT_DOCUMENT" in codes(res)


def test_cp1251_is_decoded(broken_files):
    res = run_preflight(broken_files["cp1251.md"])
    assert res.ok
    assert res.health.encoding.lower().replace("_", "-") in {"cp1251", "windows-1251"}
    assert "Ф-1" in res.text and "быстро" in res.text
    doc = parse_document(res)
    assert any("Ф-1" in b.text for b in doc.blocks)


def test_utf16_with_bom(broken_files):
    res = run_preflight(broken_files["utf16.txt"])
    assert res.ok and res.health.encoding.startswith("utf-16")
    assert "Функциональные" in res.text


def test_truncated_pdf(broken_files):
    res = run_preflight(broken_files["truncated.pdf"])
    c = codes(res)
    assert "PDF_TRUNCATED" in c
    # усечённый на 2/3 файл не открывается или не содержит текста → без traceback, статус фатальный
    assert res.health.status == "UNPROCESSABLE"


def test_encrypted_pdf(broken_files):
    res = run_preflight(broken_files["encrypted.pdf"])
    assert res.health.status == "UNPROCESSABLE"
    assert "PDF_ENCRYPTED" in codes(res)


def test_scanned_pdf_without_vision(broken_files):
    res = run_preflight(broken_files["scanned.pdf"], vision_available=False)
    assert res.health.status == "UNPROCESSABLE"
    assert "PDF_SCANNED" in codes(res)
    assert res.health.scanned_pages == [1]


def test_scanned_pdf_with_vision_is_degraded(broken_files):
    res = run_preflight(broken_files["scanned.pdf"], vision_available=True)
    assert res.health.status == "DEGRADED"


def test_blank_pdf(broken_files):
    res = run_preflight(broken_files["blank.pdf"])
    assert res.health.status == "UNPROCESSABLE"
    assert "EMPTY_TEXT" in codes(res)


def test_good_pdf_parses(broken_files):
    res = run_preflight(broken_files["good.pdf"])
    assert res.ok, res.health.issues
    doc = parse_document(res)
    assert doc.text.strip()
    assert all(b.page == 1 for b in doc.blocks)


def test_random_garbage(broken_files):
    res = run_preflight(broken_files["garbage.md"])
    assert res.health.status == "UNPROCESSABLE"


def test_fake_and_non_docx_zip(broken_files):
    assert run_preflight(broken_files["fake.docx"]).health.status == "UNPROCESSABLE"
    res = run_preflight(broken_files["zip_not_docx.docx"])
    assert "ZIP_NOT_DOCX" in codes(res)


def test_whitespace_only(broken_files):
    res = run_preflight(broken_files["whitespace.md"])
    assert "EMPTY_TEXT" in codes(res)


def test_missing_file(tmp_path):
    res = run_preflight(tmp_path / "nope.md")
    assert "FILE_NOT_FOUND" in codes(res)


def test_extension_mismatch_pdf_as_md(broken_files, tmp_path):
    p = tmp_path / "really_pdf.md"
    p.write_bytes(broken_files["good.pdf"].read_bytes())
    res = run_preflight(p)
    assert res.health.detected_format == "pdf"
    assert "EXTENSION_MISMATCH" in codes(res)


def test_docx_order_preserved(docx_file):
    res = run_preflight(docx_file)
    assert res.ok
    doc = parse_document(res)
    kinds = [(b.kind, b.text[:10]) for b in doc.blocks]
    texts = [b.text for b in doc.blocks]
    i1 = next(i for i, t in enumerate(texts) if t.startswith("Ф-1"))
    it = next(i for i, b in enumerate(doc.blocks) if b.kind == "table_row")
    i2 = next(i for i, t in enumerate(texts) if t.startswith("Ф-2"))
    assert i1 < it < i2, kinds
