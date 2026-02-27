from __future__ import annotations

from pathlib import Path


def _display_title(pdf_path: str, source_name: str | None = None) -> str:
    if source_name:
        return Path(source_name).stem
    return Path(pdf_path).stem


def _parse_with_marker(pdf_path: str, source_name: str | None = None) -> tuple[str, str, dict]:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    models = load_all_models()
    full_text, _meta = convert_single_pdf(pdf_path, models)
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError("Marker returned empty markdown content")
    return full_text, "marker", _meta


def _parse_with_pypdf(pdf_path: str, source_name: str | None = None) -> tuple[str, str, dict]:
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
    
    meta = {
        "title": title,
        "total_pages": len(reader.pages),
        "parser": "pypdf"
    }
    return markdown_text, "pypdf", meta


def _parse_with_pypdf2(pdf_path: str, source_name: str | None = None) -> tuple[str, str, dict]:
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
        
    meta = {
        "title": title,
        "total_pages": len(reader.pages),
        "parser": "PyPDF2"
    }
    return markdown_text, "PyPDF2", meta


def _parse_with_pymupdf4llm(pdf_path: str, source_name: str | None = None) -> tuple[str, str, dict]:
    import pymupdf4llm
    import fitz

    # pymupdf4llm extracts markdown preserving tables and headers
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # Get basic page count using fitz
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    if not md_text or not md_text.strip():
        raise ValueError("pymupdf4llm returned empty markdown content")
        
    # Return placeholder metadata. It will be standardized in parse_pdf_to_markdown.
    meta = {
        "title": _display_title(pdf_path, source_name),
        "total_pages": total_pages,
        "parser_engine": "pymupdf4llm"
    }
    
    return md_text.strip(), "pymupdf4llm", meta


def parse_pdf_to_markdown(
    pdf_path: str, 
    source_name: str | None = None,
    parser_choice: str = "marker",
    creator: str = "admin"
) -> tuple[str, str, dict]:
    """
    Parse PDF into markdown-like text.
    Standardizes output metadata dictionary across all parsing engines.
    """
    errors: list[str] = []
    
    # Order of parsers to try based on user choice
    parsers = []
    if parser_choice.lower() == "pymupdf4llm":
        parsers = [_parse_with_pymupdf4llm, _parse_with_marker, _parse_with_pypdf, _parse_with_pypdf2]
    else:
        parsers = [_parse_with_marker, _parse_with_pymupdf4llm, _parse_with_pypdf, _parse_with_pypdf2]

    selected_text = None
    selected_parser = None
    selected_meta = {}
    
    for parser in parsers:
        try:
            full_text, p_used, raw_meta = parser(pdf_path, source_name)
            selected_text = full_text
            selected_parser = p_used
            
            # Try to grab total pages if available from raw_meta
            total_pages = 0
            if "total_pages" in raw_meta:
                total_pages = raw_meta["total_pages"]
            elif "pages" in raw_meta:
                total_pages = raw_meta["pages"]
                
            selected_meta = raw_meta
            selected_meta["extracted_pages"] = total_pages
            break
        except ImportError as e:
            errors.append(f"{parser.__name__}: import error ({e})")
        except Exception as e:
            errors.append(f"{parser.__name__}: runtime error ({e})")

    if selected_text is None:
        raise RuntimeError(
            "No available PDF parser succeeded. "
            + " | ".join(errors[:3])
        )

    # Standardize Metadata JSON schema for RAG pipeline
    from datetime import datetime, timezone
    
    iso_time = datetime.now(timezone.utc).isoformat()
    standard_meta = {
        "source_file": source_name or Path(pdf_path).name,
        "parser_used": selected_parser,
        "total_pages": selected_meta.get("extracted_pages", 0),
        "creator": creator,
        "created_at": iso_time,
        # Preserve original metadata attributes for backward compatibility with marker format
        "languages": selected_meta.get("languages", ["ko", "en"]),
        "filetype": "pdf"
    }
    
    return selected_text, selected_parser, standard_meta
