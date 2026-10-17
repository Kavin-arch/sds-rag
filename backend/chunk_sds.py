"""
chunk_sds.py

Step 5 of the SDS RAG assistant build: turn one messy SDS PDF into a list of
clean chunks, each tagged with which of the 16 GHS sections it belongs to.

Usage:
    python chunk_sds.py path/to/acetone_sds.pdf

This will:
  1. Extract all text from the PDF with PyMuPDF
  2. Find where each of the 16 GHS sections starts using regex
  3. Slice the text between consecutive section headers
  4. Print out each chunk (document, section_number, section_title, text)

Once this looks right on your sample PDFs, step 6 wires this into
Gemini embeddings + Chroma storage (see embed_and_store() stub at the bottom).
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# The 16 standard GHS Safety Data Sheet sections, in order.
# Keys are the canonical section numbers; values are lists of words/phrases
# that commonly appear in that section's heading (used to build the regex).
# ---------------------------------------------------------------------------
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


def extract_text(pdf_path: str) -> str:
    """Pull all text out of the PDF as one long string."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def build_section_pattern() -> re.Pattern:
    """
    Build a regex that matches SDS section headers in their common forms:
        "Section 4"      "SECTION 4:"      "4. First-Aid Measures"
        "4.  First Aid Measures"           "SECTION 4 - FIRST AID MEASURES"

    We match on the number (1-16) at a line-ish boundary, optionally preceded
    by the word "SECTION" and optionally followed by a colon/dash and the
    heading text. Matching primarily on number + position is more reliable
    than matching on heading wording, since wording varies a lot between
    manufacturers.
    """
    # Matches: start of line, optional "SECTION", number 1-16, then a
    # separator (colon, dash, period, or just whitespace) before the title.
    pattern = re.compile(
        r"(?im)^\s*(?:SECTION\s*)?(\d{1,2})\s*[:\.\-]?\s*([A-Za-z][A-Za-z /,\-\(\)]{3,60})?\s*$",
    )
    return pattern


def find_section_starts(text: str):
    """
    Scan the text for lines that look like section headers, and keep only
    the ones whose number is 1-16 and appear in increasing order (this
    filters out stray numbers elsewhere in the document, like "Section 4"
    being referenced inside Section 9's text).
    """
    pattern = build_section_pattern()
    candidates = []
    for match in pattern.finditer(text):
        num = int(match.group(1))
        if 1 <= num <= 16:
            candidates.append((num, match.start(), match.group(0).strip()))

    # Keep only strictly increasing section numbers, in order of appearance,
    # so we don't pick up incidental mentions of "Section 4" inside other
    # sections' body text.
    filtered = []
    expected_next = 1
    for num, pos, raw in candidates:
        if num == expected_next:
            filtered.append((num, pos, raw))
            expected_next += 1
        elif num == expected_next + 1:
            # tolerate a missed section (e.g. bad OCR) and resync
            filtered.append((num, pos, raw))
            expected_next = num + 1

    return filtered


def chunk_sds(pdf_path: str) -> list[dict]:
    """
    Main entry point: given a path to an SDS PDF, return a list of chunk
    dicts ready to be embedded and stored in Chroma.
    """
    text = extract_text(pdf_path)
    section_starts = find_section_starts(text)

    if not section_starts:
        raise ValueError(
            f"No GHS section headers detected in {pdf_path}. "
            "This PDF may need manual chunking, or its headers use an "
            "unusual format — inspect the raw text and adjust the regex."
        )

    doc_name = Path(pdf_path).name
    chunks = []

    for i, (num, pos, raw_header) in enumerate(section_starts):
        start = pos
        end = section_starts[i + 1][1] if i + 1 < len(section_starts) else len(text)
        section_text = text[start:end].strip()

        chunks.append({
            "document": doc_name,
            "section_number": num,
            "section_title": GHS_SECTIONS.get(num, "Unknown"),
            "text": section_text,
        })

    return chunks


# ---------------------------------------------------------------------------
# Step 6 preview: once chunks look right, this is how you'd embed + store
# them in Chroma. Left as a stub here — wire it in once step 5 is verified.
# ---------------------------------------------------------------------------
def embed_and_store(chunks: list[dict], collection):
    """
    import google.generativeai as genai

    for chunk in chunks:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk["text"],
        )
        embedding = result["embedding"]

        chunk_id = f"{chunk['document']}_sec{chunk['section_number']}"
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "document": chunk["document"],
                "section": chunk["section_number"],
                "title": chunk["section_title"],
            }],
        )
    """
    raise NotImplementedError("Wire this up in step 6, after step 5 checks out.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunk_sds.py path/to/some_sds.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chunks = chunk_sds(pdf_path)

    print(f"\nFound {len(chunks)} section(s) in {Path(pdf_path).name}:\n")
    for c in chunks:
        preview = c["text"][:200].replace("\n", " ")
        print(f"--- Section {c['section_number']}: {c['section_title']} ---")
        print(f"{preview}...\n")