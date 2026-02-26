from __future__ import annotations

from pathlib import Path


def _display_title(pdf_path: str, source_name: str | None = None) -> str:
    if source_name:
        return Path(source_name).stem
    return Path(pdf_path).stem


def _parse_with_marker(pdf_path: str, source_name: str | None = None) -> tuple[str, str]:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    models = load_all_models()
    full_text, _meta = convert_single_pdf(pdf_path, models)
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError("Marker returned empty markdown content")
    return full_text, "marker"


def _parse_with_pypdf(pdf_path: str, source_name: str | None = None) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    title = _display_title(pdf_path, source_name)
    parts = [f"# {title}"]
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        parts.append(f"## Page {idx}\n\n{text}")
    markdown_text = "\n\n".join(parts).strip()
    if not markdown_text:
        raise ValueError("pypdf returned no extractable text")
    return markdown_text, "pypdf"


def _parse_with_pypdf2(pdf_path: str, source_name: str | None = None) -> tuple[str, str]:
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    title = _display_title(pdf_path, source_name)
    parts = [f"# {title}"]
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        parts.append(f"## Page {idx}\n\n{text}")
    markdown_text = "\n\n".join(parts).strip()
    if not markdown_text:
        raise ValueError("PyPDF2 returned no extractable text")
    return markdown_text, "PyPDF2"


def parse_pdf_to_markdown(pdf_path: str, source_name: str | None = None) -> tuple[str, str]:
    """
    Parse PDF into markdown-like text.

    Preference order:
    1) marker-pdf (if installed)
    2) pypdf
    3) PyPDF2
    """
    errors: list[str] = []

    for parser in (_parse_with_marker, _parse_with_pypdf, _parse_with_pypdf2):
        try:
            return parser(pdf_path, source_name)
        except ImportError as e:
            errors.append(f"{parser.__name__}: import error ({e})")
        except Exception as e:
            errors.append(f"{parser.__name__}: runtime error ({e})")

    raise RuntimeError(
        "No available PDF parser. Install one of: marker-pdf (separate env recommended), pypdf, PyPDF2. "
        + " | ".join(errors[:3])
    )
