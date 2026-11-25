"""Preflight Gatekeeper: проверка «здоровья» входного файла ДО парсинга и LLM.

Принцип: никогда не падать с traceback. Любая проблема превращается в
``HealthIssue`` с кодом, понятным сообщением и рекомендацией на русском.
Если файл нельзя обработать — статус ``UNPROCESSABLE``, LLM не вызывается,
а CLI пишет диагностический отчёт и возвращает exit code 2.
"""

from __future__ import annotations

import codecs
import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .models import DocumentHealthReport

MAX_BYTES = 50 * 1024 * 1024
MIN_TEXT_CHARS = 20
SCAN_TEXT_CHARS = 60  # меньше текста на странице => кандидат в скан
SCAN_IMAGE_COVERAGE = 0.45  # доля площади страницы, покрытая растром
DIAGRAM_MIN_DRAWINGS = 120

TEXT_EXTS = {".md", ".markdown", ".txt", ".text", ".rst"}
KNOWN_EXTS = TEXT_EXTS | {".pdf", ".docx"}

_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "PNG-изображение"),
    (b"\xff\xd8\xff", "JPEG-изображение"),
    (b"GIF87a", "GIF-изображение"),
    (b"GIF89a", "GIF-изображение"),
    (b"BM", "BMP-изображение"),
    (b"II*\x00", "TIFF-изображение"),
    (b"MM\x00*", "TIFF-изображение"),
    (b"RIFF", "RIFF/WebP-контейнер"),
]
_EXEC_MAGIC: list[tuple[bytes, str]] = [
    (b"MZ", "исполняемый файл Windows"),
    (b"\x7fELF", "исполняемый файл ELF"),
    (b"\x1f\x8b", "gzip-архив"),
    (b"Rar!", "RAR-архив"),
    (b"7z\xbc\xaf\x27\x1c", "7z-архив"),
]
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@dataclass
class PreflightResult:
    health: DocumentHealthReport
    data: bytes = b""
    text: str | None = None  # для текстовых форматов — уже декодированный текст
    page_flags: dict[int, str] = field(default_factory=dict)  # page -> "scan" | "diagram"

    @property
    def ok(self) -> bool:
        return self.health.status != "UNPROCESSABLE"


# ---------------------------------------------------------------------------


def _sniff(data: bytes) -> tuple[str, str]:
    """Определяет реальный формат по сигнатуре. Возвращает (format, описание)."""
    head = data[:1024]
    if b"%PDF" in head:
        return "pdf", "PDF"
    if data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06"):
        return "zip", "ZIP-контейнер"
    if data.startswith(OLE_MAGIC):
        return "ole", "OLE2-контейнер (DOC/XLS или зашифрованный OOXML)"
    for magic, name in _IMAGE_MAGIC:
        if data.startswith(magic):
            if magic == b"RIFF" and data[8:12] != b"WEBP":
                continue
            if magic == b"BM" and len(data) > 14 and not _looks_like_bmp(data):
                continue
            return "image", name
    for magic, name in _EXEC_MAGIC:
        if data.startswith(magic):
            if magic == b"MZ" and b"This program cannot be run" not in data[:512] and b"PE\x00\x00" not in data[:1024]:
                continue
            return "binary", name
    return "text?", "текст"


def _looks_like_bmp(data: bytes) -> bool:
    size = int.from_bytes(data[2:6], "little")
    return abs(size - len(data)) < 1024


def _decode_text(data: bytes, health: DocumentHealthReport) -> str | None:
    """Декодирует текстовый файл: BOM → строгий UTF-8 → charset_normalizer (cp1251 и др.)."""
    from charset_normalizer import from_bytes, is_binary

    for bom, enc in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if data.startswith(bom):
            try:
                text = data.decode(enc)
                health.encoding = enc
                return text
            except UnicodeDecodeError:
                break

    try:
        text = data.decode("utf-8")
        health.encoding = "utf-8"
        if "\x00" in text:
            raise UnicodeDecodeError("utf-8", data, 0, 1, "NUL bytes")
        return text
    except UnicodeDecodeError:
        pass

    # UTF-16 без BOM: много нулевых байтов на чётных/нечётных позициях
    sample = data[:4000]
    if sample and sample.count(b"\x00") / len(sample) > 0.3:
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                text = data.decode(enc)
                if sum(ch.isprintable() or ch.isspace() for ch in text[:2000]) / max(1, len(text[:2000])) > 0.95:
                    health.encoding = enc
                    return text
            except UnicodeDecodeError:
                continue

    try:
        binary = is_binary(data)
    except Exception:  # pragma: no cover - защитный контур
        binary = False
    if binary:
        health.add(
            "BINARY_CONTENT",
            "fatal",
            "Файл содержит двоичные данные, а не текст требований (вероятно, подмена расширения или повреждение).",
            "Передайте исходный документ в формате .md/.txt (UTF-8), .docx или .pdf с текстовым слоем.",
        )
        return None

    best = from_bytes(data).best()
    if best is None:
        health.add(
            "UNKNOWN_ENCODING",
            "fatal",
            "Не удалось определить кодировку текстового файла.",
            "Пересохраните файл в UTF-8.",
        )
        return None
    health.encoding = best.encoding
    health.add(
        "NON_UTF8_ENCODING",
        "info",
        f"Файл в кодировке {best.encoding}; выполнено автоматическое перекодирование в UTF-8.",
        "Рекомендуется хранить требования в UTF-8.",
    )
    return str(best)


def _check_text_quality(text: str, health: DocumentHealthReport) -> None:
    stripped = text.strip()
    health.text_chars = len(stripped)
    if len(stripped) < MIN_TEXT_CHARS:
        health.add(
            "EMPTY_TEXT",
            "fatal",
            f"В документе нет содержательного текста ({len(stripped)} симв.).",
            "Проверьте, что выгружен правильный файл и он не пустой.",
        )
        return
    bad = sum(1 for ch in stripped if ch == "�" or (ord(ch) < 32 and ch not in "\n\r\t\f"))
    ratio = bad / len(stripped)
    if ratio > 0.10:
        health.add(
            "GARBLED_TEXT",
            "fatal",
            f"Текст повреждён: {ratio:.0%} недопустимых символов (кракозябры / бинарный мусор).",
            "Пересохраните документ в UTF-8 или выгрузите заново из исходной системы.",
        )
    elif ratio > 0.01:
        health.add(
            "SUSPICIOUS_CHARS",
            "warning",
            f"Обнаружено {bad} недопустимых символов ({ratio:.1%}); часть цитат может быть искажена.",
            "Проверьте кодировку исходного файла.",
        )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _check_pdf(data: bytes, res: PreflightResult, vision_available: bool) -> None:
    health = res.health
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        health.add("PDF_LIB_MISSING", "fatal", "Не установлен pymupdf.", "pip install pymupdf")
        return

    if b"%%EOF" not in data[-2048:]:
        health.add(
            "PDF_TRUNCATED",
            "warning",
            "PDF не содержит маркера конца файла %%EOF — файл, вероятно, усечён при загрузке.",
            "Скачайте документ заново; результаты могут быть неполными.",
        )

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.EmptyFileError:
        health.add("PDF_EMPTY", "fatal", "PDF-файл пуст.", "Выгрузите документ заново.")
        return
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        health.add(
            "PDF_CORRUPTED",
            "fatal",
            f"PDF повреждён и не может быть открыт: {exc}".strip(),
            "Скачайте документ заново или пересохраните его из исходного редактора («Печать в PDF»).",
        )
        return

    with doc:
        if doc.needs_pass:
            if not doc.authenticate(""):
                health.add(
                    "PDF_ENCRYPTED",
                    "fatal",
                    "PDF защищён паролем на открытие — текст недоступен.",
                    "Снимите защиту (или передайте незащищённую копию) и повторите прогон.",
                )
                return
        if doc.is_encrypted:
            health.add(
                "PDF_OWNER_PROTECTED",
                "info",
                "PDF зашифрован с правами владельца (без пароля на открытие); текст извлечён.",
            )
        health.pages = doc.page_count
        if doc.page_count == 0:
            health.add("PDF_NO_PAGES", "fatal", "В PDF нет страниц.", "Выгрузите документ заново.")
            return

        total_chars = 0
        for pno in range(doc.page_count):
            try:
                page = doc[pno]
                text = page.get_text("text") or ""
            except Exception as exc:  # повреждённая страница
                health.add(
                    "PDF_PAGE_ERROR",
                    "warning",
                    f"Страница {pno + 1} не читается: {exc}",
                    "Проверьте эту страницу вручную.",
                )
                continue
            chars = len(text.strip())
            total_chars += chars
            area = abs(page.rect) or 1.0
            img_area = 0.0
            try:
                for info in page.get_image_info():
                    x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
                    img_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
            except Exception:
                pass
            coverage = min(1.0, img_area / area)
            n_draw = 0
            try:
                n_draw = len(page.get_drawings())
            except Exception:
                pass
            if chars < SCAN_TEXT_CHARS and coverage >= SCAN_IMAGE_COVERAGE:
                health.scanned_pages.append(pno + 1)
                res.page_flags[pno + 1] = "scan"
            elif n_draw >= DIAGRAM_MIN_DRAWINGS or (coverage >= 0.25 and chars < 400):
                health.diagram_pages.append(pno + 1)
                res.page_flags[pno + 1] = "diagram"

        health.text_chars = total_chars
        n = doc.page_count
        if health.scanned_pages:
            if len(health.scanned_pages) == n and total_chars < MIN_TEXT_CHARS:
                if vision_available:
                    health.add(
                        "PDF_SCANNED",
                        "warning",
                        f"PDF — скан без текстового слоя ({n} стр.). Текст будет распознан через Vision-модель; "
                        "точность цитат ниже.",
                        "Для максимальной точности передайте исходный текстовый документ.",
                    )
                else:
                    health.add(
                        "PDF_SCANNED",
                        "fatal",
                        f"PDF — скан без текстового слоя ({n} стр.): извлечь требования невозможно.",
                        "Передайте исходный документ (.docx/.md) или PDF с текстовым слоем; либо запустите с "
                        "LLM-ключом Gemini для Vision-распознавания.",
                    )
                return
            health.add(
                "PDF_PARTIAL_SCAN",
                "warning",
                f"Страницы-сканы без текста: {', '.join(map(str, health.scanned_pages))}.",
                "Проверьте эти страницы вручную или включите Vision-распознавание (LLM-режим).",
            )
        if health.diagram_pages:
            health.add(
                "PDF_DIAGRAMS",
                "info",
                f"Страницы с диаграммами/графикой: {', '.join(map(str, health.diagram_pages))}. "
                "Содержимое схем анализируется только по подписям.",
            )
        if total_chars < MIN_TEXT_CHARS and not health.scanned_pages:
            health.add(
                "EMPTY_TEXT",
                "fatal",
                "PDF не содержит извлекаемого текста.",
                "Передайте документ с текстовым слоем.",
            )


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def _check_docx(data: bytes, health: DocumentHealthReport) -> None:
    try:
        import docx  # python-docx

        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        health.add(
            "DOCX_CORRUPTED",
            "fatal",
            f"DOCX повреждён и не может быть открыт ({type(exc).__name__}).",
            "Откройте файл в Word/LibreOffice и пересохраните как .docx.",
        )
        return
    chars = sum(len(p.text.strip()) for p in document.paragraphs)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                chars += len(cell.text.strip())
    health.text_chars = chars
    if chars < MIN_TEXT_CHARS:
        health.add(
            "EMPTY_TEXT",
            "fatal",
            "DOCX не содержит текста (возможно, требования вставлены картинками).",
            "Передайте документ с текстом требований, а не изображениями.",
        )


# ---------------------------------------------------------------------------


def run_preflight(path: str | Path, *, vision_available: bool = False) -> PreflightResult:
    """Главная точка входа. Никогда не бросает исключения наружу."""
    p = Path(path)
    health = DocumentHealthReport(path=str(p), file_name=p.name, declared_extension=p.suffix.lower())
    res = PreflightResult(health=health)
    try:
        _run(p, res, vision_available)
    except Exception as exc:  # последний рубеж: graceful degradation
        health.add(
            "PREFLIGHT_INTERNAL_ERROR",
            "fatal",
            f"Непредвиденная ошибка при проверке файла: {type(exc).__name__}: {exc}",
            "Сообщите разработчикам агента и приложите файл.",
        )
    return res


def _run(p: Path, res: PreflightResult, vision_available: bool) -> None:
    health = res.health
    if not p.exists():
        health.add("FILE_NOT_FOUND", "fatal", f"Файл не найден: {p}", "Проверьте путь к документу.")
        return
    if not p.is_file():
        health.add("NOT_A_FILE", "fatal", f"Путь не является файлом: {p}", "Укажите путь к файлу документа.")
        return
    size = p.stat().st_size
    health.size_bytes = size
    if size == 0:
        health.add(
            "EMPTY_FILE",
            "fatal",
            "Файл пустой (0 байт).",
            "Выгрузите документ заново — вероятно, загрузка прервалась.",
        )
        return
    if size > MAX_BYTES:
        health.add(
            "FILE_TOO_LARGE",
            "fatal",
            f"Файл слишком большой ({size / 1024 / 1024:.1f} МБ > {MAX_BYTES // 1024 // 1024} МБ).",
            "Разделите документ на части.",
        )
        return
    try:
        data = p.read_bytes()
    except OSError as exc:
        health.add("READ_ERROR", "fatal", f"Не удалось прочитать файл: {exc}", "Проверьте права доступа.")
        return
    res.data = data
    health.sha256 = hashlib.sha256(data).hexdigest()

    kind, kind_name = _sniff(data)
    ext = health.declared_extension

    if kind == "image":
        health.detected_format = "unknown"
        health.add(
            "IMAGE_NOT_DOCUMENT",
            "fatal",
            f"Файл с расширением «{ext or '—'}» на самом деле является {kind_name}.",
            "Передайте текстовый документ требований (.md/.docx/.pdf). Скриншоты не поддерживаются.",
        )
        return
    if kind == "binary":
        health.add(
            "BINARY_FILE",
            "fatal",
            f"Файл является «{kind_name}», а не документом требований.",
            "Проверьте, что выбран правильный файл.",
        )
        return
    if kind == "ole":
        if b"E\x00n\x00c\x00r\x00y\x00p\x00t\x00e\x00d\x00P\x00a\x00c\x00k\x00a\x00g\x00e" in data:
            health.add(
                "DOCX_ENCRYPTED",
                "fatal",
                "Документ Office зашифрован паролем.",
                "Снимите пароль в Word («Файл → Сведения → Защита документа») и повторите прогон.",
            )
        else:
            health.add(
                "LEGACY_DOC",
                "fatal",
                "Формат .doc (Word 97–2003) не поддерживается.",
                "Пересохраните документ как .docx или .pdf.",
            )
        return
    if kind == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            health.add(
                "ZIP_CORRUPTED",
                "fatal",
                "Файл похож на DOCX (ZIP-контейнер), но архив повреждён.",
                "Пересохраните документ в Word/LibreOffice.",
            )
            return
        if "[Content_Types].xml" not in names or not any(n.startswith("word/") for n in names):
            health.add(
                "ZIP_NOT_DOCX",
                "fatal",
                "ZIP-контейнер не является документом Word (нет [Content_Types].xml или word/).",
                "Передайте файл .docx, .pdf или .md.",
            )
            return
        kind = "docx"

    if kind == "pdf":
        health.detected_format = "pdf"
    elif kind == "docx":
        health.detected_format = "docx"
    else:
        health.detected_format = "md" if ext in {".md", ".markdown"} else "txt"

    if ext and ext not in KNOWN_EXTS:
        health.add(
            "UNKNOWN_EXTENSION",
            "info",
            f"Расширение «{ext}» не входит в список поддерживаемых; формат определён по содержимому: {health.detected_format}.",
        )
    elif ext and ((ext == ".pdf") != (kind == "pdf") or (ext == ".docx") != (kind == "docx")):
        health.add(
            "EXTENSION_MISMATCH",
            "warning",
            f"Расширение «{ext}» не соответствует содержимому ({kind_name}); "
            f"файл обработан как {health.detected_format}.",
            "Переименуйте файл в соответствии с его реальным форматом.",
        )

    if kind == "pdf":
        _check_pdf(data, res, vision_available)
    elif kind == "docx":
        _check_docx(data, health)
    else:
        text = _decode_text(data, health)
        if text is None:
            return
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        res.text = text
        _check_text_quality(text, health)
