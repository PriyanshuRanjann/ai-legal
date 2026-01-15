from pathlib import Path
from typing import List, Sequence
from .schemas import PDFPageSchema
import fitz
import json
from pydantic import BaseModel, Field, validator


def parse_pdf(path: str | Path, document_id: str | None = None) -> List[PDFPageSchema]:
    """Return per-page text plus metadata using PyMuPDF only."""
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = document_id or pdf_path.stem.lower().replace(" ", "-")
    pages: List[PDFPageSchema] = []

    try:
        with fitz.open(pdf_path) as doc:
            for index, page in enumerate(doc):
                try:
                    raw_text = page.get_text("text")
                except Exception as extraction_error:
                    raise RuntimeError(
                        f"Unable to extract text from {pdf_path.name} (page index {index})"
                    ) from extraction_error

                cleaned = _normalize_text(raw_text)
                pages.append(
                    PDFPageSchema(
                        document_id=doc_id,
                        page_number=index,
                        text=cleaned,
                        source_path=str(pdf_path),
                    )
                )
    except Exception as open_error:
        raise RuntimeError(f"Failed to parse PDF {pdf_path}") from open_error

    return pages


def _normalize_text(raw_text: str | None) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(part for part in text.split() if part)
    return text.strip()


def save_pdf_pages_to_json(
    pdf_path: str | Path, output_dir: str | Path = "output/parsed_pdf"
) -> None:
    """Parse PDF and save each page as a JSON file in output_dir."""
    pages = parse_pdf(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for page in pages:
        output_path = output_dir / f"{page.document_id}_page_{page.page_number}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(page.model_dump(), f, ensure_ascii=False, indent=4)
        print(
            f"Saved page {page.page_number} ({page.char_count} chars) to {output_path}"
        )


if __name__ == "__main__":
    # Single function call: input PDF, output JSONs
    input = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    save_pdf_pages_to_json(input)
