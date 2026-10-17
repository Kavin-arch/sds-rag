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
   .\venv\Scripts\Activate.ps1
   pip install -r backend\requirements.txt
   uvicorn main:app --app-dir backend --reload
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

Already indexed section IDs are skipped. The frontend library lists documents represented in the Chroma collection; the PDF files in `sds/` are not automatically indexed on startup.
