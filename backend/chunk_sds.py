"""Extract SDS PDFs into page-aware, searchable GHS section passages."""

import re
import sys
from pathlib import Path

import fitz

GHS_SECTIONS = {
    1: "Identification",
    2: "Hazard(s) Identification",
    3: "Composition/Information on Ingredients",
    4: "First-Aid Measures",
    5: "Fire-Fighting Measures",
    6: "Accidental Release Measures",
    7: "Handling and Storage",
    8: "Exposure Controls/Personal Protection",
    9: "Physical and Chemical Properties",
    10: "Stability and Reactivity",
    11: "Toxicological Information",
    12: "Ecological Information",
    13: "Disposal Considerations",
    14: "Transport Information",
    15: "Regulatory Information",
    16: "Other Information",
}

# Headings must include a title, avoiding accidental matches on numbered body text.
HEADER_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:SECTION[ \t]+)?(1[0-6]|[1-9])"
    r"[ \t]*(?:[:.)\-][ \t]*|[ \t]+)"
    r"([A-Z][A-Z0-9 /,&()'\-]{2,90})[ \t]*$"
)
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 220


def extract_pages(pdf_path: str) -> list[str]:
    """Extract text page by page and fail clearly when OCR is needed."""
    with fitz.open(pdf_path) as pdf:
        pages = [page.get_text("text", sort=True).strip() for page in pdf]
    if not any(pages):
        raise ValueError(
            "No selectable text was found in this PDF. It may be scanned; run OCR on it before uploading."
        )
    return pages


def _split_passages(text: str) -> list[tuple[int, int, str]]:
    """Split long section text near paragraph/sentence boundaries with overlap."""
    passages = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start + MAX_CHUNK_CHARS // 2, end),
                text.rfind("\n", start + MAX_CHUNK_CHARS // 2, end),
                text.rfind(". ", start + MAX_CHUNK_CHARS // 2, end),
            )
            if boundary > start:
                end = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
        value = text[start:end].strip()
        if value:
            passages.append((start, end, value))
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return passages


def chunk_sds(pdf_path: str) -> list[dict]:
    """Return page-aware chunks with canonical GHS metadata."""
    pages = extract_pages(pdf_path)
    page_offsets: list[tuple[int, int, int]] = []
    parts = []
    offset = 0
    for page_number, page_text in enumerate(pages, start=1):
        page_offsets.append((offset, offset + len(page_text), page_number))
        parts.append(page_text)
        offset += len(page_text) + 2
    full_text = "\n\n".join(parts)

    starts = []
    expected = 1
    for match in HEADER_PATTERN.finditer(full_text):
        number = int(match.group(1))
        if number == expected or number == expected + 1:
            starts.append((number, match.start()))
            expected = number + 1
    if not starts:
        raise ValueError(
            "No recognizable GHS section headings were found. Check the PDF text or OCR it before uploading."
        )

    filename = Path(pdf_path).name
    chunks = []
    for section_index, (number, section_start) in enumerate(starts):
        section_end = starts[section_index + 1][1] if section_index + 1 < len(starts) else len(full_text)
        section_text = full_text[section_start:section_end].strip()
        # Skip page separators from the stored passage while retaining absolute source offsets.
        for passage_index, (relative_start, relative_end, passage) in enumerate(_split_passages(section_text)):
            absolute_start = section_start + relative_start
            absolute_end = section_start + relative_end
            page_start = next((page for begin, end, page in page_offsets if begin <= absolute_start <= end), 1)
            page_end = next((page for begin, end, page in page_offsets if begin <= absolute_end <= end), page_start)
            chunks.append({
                "document": filename,
                "section_number": number,
                "section_title": GHS_SECTIONS[number],
                "chunk_index": passage_index,
                "page_start": page_start,
                "page_end": max(page_start, page_end),
                "text": passage,
            })
    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python chunk_sds.py path/to/file.pdf")
    result = chunk_sds(sys.argv[1])
    print(f"Found {len(result)} passage(s) in {Path(sys.argv[1]).name}:")
    for chunk in result:
        print(
            f"Section {chunk['section_number']} ({chunk['section_title']}), "
            f"pages {chunk['page_start']}-{chunk['page_end']}: {chunk['text'][:180]}..."
        )
