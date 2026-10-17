"""
embed_and_store.py

Step 6 of the SDS RAG assistant build: take the chunks produced by
chunk_sds.py, embed each one with Gemini's text-embedding-004 model,
and store them in a local, persistent Chroma collection.

Usage:
    python embed_and_store.py sds/acetone_sds.pdf
    python embed_and_store.py sds/           # embed every PDF in a folder

Requires:
    - GEMINI_API_KEY set in your .env file (see Step 4 of the guide)
    - chunk_sds.py in the same folder (this script imports chunk_sds from it)
"""

import os
import sys
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

from chunk_sds import chunk_sds

# ---------------------------------------------------------------------------
# Setup: load API key, configure Gemini, open (or create) the Chroma store
#
# NOTE: the old `google-generativeai` package is fully deprecated, and its
# `text-embedding-004` model has been discontinued. This script uses the
# current `google-genai` package and the `gemini-embedding-001` model.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # reads GEMINI_API_KEY from project .env

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Make sure you have a .env file in this "
        "folder containing: GEMINI_API_KEY=your_key_here"
    )

gemini_client = genai.Client(api_key=api_key)

# PersistentClient means Chroma saves to disk (./chroma_db) so your
# embeddings survive a restart — you don't have to re-embed every run.
db_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
collection = db_client.get_or_create_collection("sds_sections")

EMBED_MODEL = "gemini-embedding-001"


def embed_text(text: str) -> list[float]:
    """Send one chunk of text to Gemini's embedding model, get back a vector."""
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


def store_chunks(chunks: list[dict]):
    """Embed each chunk and add it to the Chroma collection."""
    for chunk in chunks:
        chunk_id = f"{chunk['document']}_sec{chunk['section_number']}"

        # Skip re-embedding if this exact chunk is already stored — saves
        # API calls if you re-run the script on the same PDF.
        existing = collection.get(ids=[chunk_id])
        if existing["ids"]:
            print(f"  [skip] {chunk_id} already in the collection")
            continue

        embedding = embed_text(chunk["text"])

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
        print(f"  [stored] {chunk_id} ({chunk['section_title']})")


def process_pdf(pdf_path: str):
    print(f"\nProcessing {pdf_path} ...")
    chunks = chunk_sds(pdf_path)
    print(f"  {len(chunks)} section(s) found, embedding + storing now:")
    store_chunks(chunks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embed_and_store.py path/to/file.pdf")
        print("       python embed_and_store.py path/to/folder/")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        pdf_files = sorted(target.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {target}")
            sys.exit(1)
        for pdf_file in pdf_files:
            process_pdf(str(pdf_file))
    elif target.is_file():
        process_pdf(str(target))
    else:
        print(f"Path not found: {target}")
        sys.exit(1)

    print(f"\nDone. Collection now has {collection.count()} chunk(s) total.")
