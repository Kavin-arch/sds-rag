"""Index an SDS PDF or directory using the same v2 pipeline as the API."""

import sys
from pathlib import Path

from chunk_sds import chunk_sds
from main import collection, store_chunks


def process_pdf(pdf_path: Path) -> None:
    print(f"\nProcessing {pdf_path} ...")
    chunks = chunk_sds(str(pdf_path))
    sections = len({chunk["section_number"] for chunk in chunks})
    passages = store_chunks(chunks)
    print(f"  Indexed {passages} passages across {sections} GHS sections.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python backend/embed_and_store.py path/to/file.pdf|folder/")

    target = Path(sys.argv[1]).resolve()
    if target.is_dir():
        pdf_files = sorted(target.glob("*.pdf"))
        if not pdf_files:
            raise SystemExit(f"No PDFs found in {target}")
        for pdf_file in pdf_files:
            process_pdf(pdf_file)
    elif target.is_file() and target.suffix.lower() == ".pdf":
        process_pdf(target)
    else:
        raise SystemExit(f"PDF or folder not found: {target}")

    print(f"\nDone. V2 passage collection contains {collection.count()} passage(s).")
