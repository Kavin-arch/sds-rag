"""
main.py

Step 7 of the SDS RAG assistant build: the FastAPI app itself.

Endpoints:
    POST /upload   - upload a new SDS PDF, chunk it, embed it, store it
    POST /chat     - ask a question, get an answer grounded in your SDS docs
    GET  /summary/{document} - get all chunks for one document (step 8 preview)

Run with:
    uvicorn main:app --reload

Then test at:
    http://127.0.0.1:8000/docs   (FastAPI's built-in interactive test UI)
"""

import os
import shutil
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

from chunk_sds import chunk_sds

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Make sure your .env file contains: "
        "GEMINI_API_KEY=your_key_here"
    )

gemini_client = genai.Client(api_key=api_key)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"  # fast + cheap, good for this kind of Q&A

db_client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
collection = db_client.get_or_create_collection("sds_sections")

UPLOAD_DIR = PROJECT_ROOT / "sds"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SDS Intelligence", version="1.0.0")


class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed_text(text: str, task_type: str) -> list[float]:
    """
    task_type should be "RETRIEVAL_DOCUMENT" when embedding chunks to store,
    and "RETRIEVAL_QUERY" when embedding a user's question to search with.
    Using the right task_type for each improves retrieval quality.
    """
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values


def store_chunks(chunks: list[dict]):
    for chunk in chunks:
        chunk_id = f"{chunk['document']}_sec{chunk['section_number']}"
        existing = collection.get(ids=[chunk_id])
        if existing["ids"]:
            continue
        embedding = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/upload")
def upload_sds(file: UploadFile = File(...)):
    """Accept a PDF upload, save it, chunk it, embed it, and store it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_filename = Path(file.filename).name
    dest_path = UPLOAD_DIR / safe_filename
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = chunk_sds(str(dest_path))
    except ValueError as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    store_chunks(chunks)

    return {
        "filename": safe_filename,
        "sections_found": len(chunks),
        "message": "Uploaded, chunked, and stored successfully.",
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """The core RAG loop: retrieve relevant chunks, then ask Gemini to answer using only them."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please enter a question.")
    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Upload an SDS PDF first.",
        )

    q_embedding = embed_text(question, task_type="RETRIEVAL_QUERY")
    results = collection.query(query_embeddings=[q_embedding], n_results=4)

    context = "\n\n".join(
        f"[{m['document']} - Section {m['section']}: {m['title']}]\n{doc}"
        for doc, m in zip(results["documents"][0], results["metadatas"][0])
    )

    prompt = f"""Answer ONLY using the context below. Always cite the document
name and section number. If the answer isn't in the context, say you
don't have that information rather than guessing.

Context:
{context}

Question: {question}"""

    response = gemini_client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    return {
        "answer": response.text or "I couldn't generate an answer. Please try asking another way.",
        "sources": [
            {"document": m["document"], "section": m["section"], "title": m["title"]}
            for m in results["metadatas"][0]
        ],
    }


@app.get("/summary/{document}")
def summary(document: str):
    """Pull every stored chunk for one document — no similarity search needed."""
    results = collection.get(where={"document": document})
    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"No stored data for '{document}'.")

    sections = sorted(
        zip(results["metadatas"], results["documents"]),
        key=lambda pair: pair[0]["section"],
    )
    return {
        "document": document,
        "sections": [
            {"section": m["section"], "title": m["title"], "text": doc}
            for m, doc in sections
        ],
    }


@app.get("/")
def root():
    return {
        "status": "running",
        "chunks_stored": collection.count(),
        "docs": "Visit /docs for the interactive API tester.",
    }


@app.get("/documents")
def documents():
    """List indexed SDS files with the number of sections stored for each."""
    results = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in results["metadatas"] or []:
        name = metadata.get("document")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return {
        "documents": [
            {"filename": name, "sections": count}
            for name, count in sorted(counts.items(), key=lambda item: item[0].lower())
        ],
        "total_sections": sum(counts.values()),
    }
