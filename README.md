# SDS Intelligence

A local Safety Data Sheet workspace: upload SDS PDFs, browse your indexed library, and ask questions grounded in the document sections stored in Chroma.

## Project layout

```text
backend/       FastAPI app, SDS parser, and indexing CLI
frontend/      React + TypeScript + Vite application
sds/           Source PDFs and uploaded PDFs
chroma_db/     Existing local Chroma database
.env           Local Gemini API key (never commit this file)
```

The existing `sds/` PDFs and `chroma_db/` database remain at the project root. The backend uses paths anchored to the project directory, so it can be started from the root or `backend/`.

## Run locally

1. Add your Gemini key to the root `.env` file:

   ```dotenv
   GEMINI_API_KEY=your_key_here
   ```

2. Install backend requirements and start FastAPI from the project root:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   .\venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --reload
   ```

   The API is at `http://127.0.0.1:8000`; its interactive docs are at `/docs`.

3. In another terminal, install and start the frontend:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

   Open the local URL printed by Vite. Its development proxy forwards `/api` requests to FastAPI. For a separately hosted API, set `VITE_API_URL` to its base URL (including any prefix) before building.

## Run with Docker

1. Copy `.env.example` to `.env` and add your Gemini API key.
2. From the project root, run:

   ```powershell
   docker compose up --build
   ```

3. Open `http://localhost:8080`. The API is also available at `http://localhost:8000`.

Compose mounts `sds/` and `chroma_db/` from the project into the backend container, so uploaded PDFs and the local index persist on your computer. Keep `.env` private; it is excluded from Git.

## Index PDFs from the CLI

From the project root, for example:

```powershell
python backend\embed_and_store.py sds\
```

The frontend library lists documents represented in Chroma; PDFs in `sds/` are not automatically indexed on startup.

## Rebuild the passage index

The current RAG pipeline uses the `sds_passages_v2` collection with smaller overlapping passages and page metadata. The previous `sds_sections` collection is left untouched. After upgrading from the original version, rebuild the passage index once:

```powershell
python backend\embed_and_store.py sds\
```

The same command can be run in Docker after starting Compose:

```powershell
docker compose exec backend python backend/embed_and_store.py /app/sds
```

Reindexing updates passages for PDFs with the same filename and removes stale passages after successful embedding. Scanned PDFs need OCR before upload; the parser reports when no selectable text is available.
