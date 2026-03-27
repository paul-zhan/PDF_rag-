# PDF RAG

A simple Retrieval-Augmented Generation (RAG) project for querying PDF documents with a local LLM stack.

The project uses:

- `Qdrant` as the vector database
- `SentenceTransformers` (`BAAI/bge-base-en-v1.5`) for embeddings
- `Ollama` for local LLM inference
- `FastAPI` for the backend API
- `Streamlit` for the frontend UI
- `Docker Compose` to run the full stack

## Architecture

The application is split into four services:

- `qdrant`: stores chunk embeddings and metadata
- `indexer`: loads PDFs from the `documents/` folder, chunks them, generates embeddings, and upserts them into Qdrant
- `backend`: receives user questions, retrieves relevant chunks from Qdrant, builds a prompt, and asks Ollama for an answer
- `frontend`: provides a simple Streamlit interface for submitting questions

## How It Works

### Indexing flow

1. PDFs are loaded recursively from the `documents/` directory.
2. Documents are split into chunks with `RecursiveCharacterTextSplitter`.
3. Chunks are embedded with `BAAI/bge-base-en-v1.5`.
4. Embeddings are stored in a Qdrant collection named `pdf_embeddings`.

### Question-answering flow

1. A user submits a question from the Streamlit UI.
2. The frontend sends the question to the FastAPI backend.
3. The backend embeds the query and retrieves the top matching chunks from Qdrant.
4. The retrieved context is injected into a prompt.
5. Ollama generates the final answer.

## Project Structure

```text
PDF_RAG/
+-- backend/
|   +-- app.py
|   +-- Dockerfile
|   +-- requirements.txt
|   +-- services/
|       +-- generation_pipeline.py
|       +-- indexing_pipeline.py
+-- frontend/
|   +-- streamlit_app.py
|   +-- Dockerfile
|   +-- requirements.txt
+-- documents/
+-- docker-compose.yml
+-- README.md
```

## Prerequisites

- Docker
- Docker Compose

Optional but recommended:

- A machine with GPU support if you want faster embedding generation

## Running The Project

### 1. Add your PDFs

Place your PDF files inside the root `documents/` directory.

### 2. Start the stack

From the project root, run:

```bash
docker compose up --build
```

This starts:

- Qdrant on `http://localhost:6333`
- Ollama on `http://localhost:11434`
- FastAPI backend on `http://localhost:8000`
- Streamlit frontend on `http://localhost:8501`

The `indexer` service runs once during startup and processes the PDFs from `documents/`.

### 3. Open the frontend

Visit:

```text
http://localhost:8501
```

Enter a question and submit it to get an answer based on the indexed PDFs.

## API

### Health check

`GET /health`

Example response:

```json
{
  "status": "200",
  "message": "OK"
}
```

### Ask a question

`POST /chatbot_answer`

Request body:

```json
{
  "query": "What does the document say about large language models?"
}
```

Response body:

```json
{
  "answer": "..."
}
```

Backend base URL:

```text
http://localhost:8000
```

## Configuration

The project reads the following environment variables:

- `QDRANT_URL`: defaults to `http://qdrant:6333`
- `OLLAMA_BASE_URL`: defaults to `http://ollama:11434`
- `OLLAMA_MODEL`: defaults to `gpt-oss`

You can override `OLLAMA_MODEL` when starting Docker Compose, for example:

```bash
$env:OLLAMA_MODEL="llama3.2"
docker compose up --build
```

## Notes

- The `documents/` folder is gitignored, so your PDFs are not committed by default.
- The embedding model dimension is currently fixed to `768` in Qdrant collection creation.
- The backend prompt instructs the model to answer only from retrieved context and reply with `I don't know` when the answer is not supported.
- The current frontend is intentionally minimal and sends requests directly to the backend service inside Docker.

## Development

### Backend

Main files:

- `backend/app.py`
- `backend/services/indexing_pipeline.py`
- `backend/services/generation_pipeline.py`

### Frontend

Main file:

- `frontend/streamlit_app.py`

## Known Limitations

- Re-indexing uses sequential numeric point IDs, so repeated indexing can overwrite previous entries instead of versioning them.
- There is no document management UI yet.
- There are no automated tests in the current repository.
- The retrieval prompt currently passes raw retrieved chunk data directly into the LLM prompt without extra citation formatting.

## Future Improvements

- Add document upload support from the frontend
- Show source citations and page numbers in responses
- Add metadata filtering in Qdrant
- Add tests for indexing, retrieval, and API routes
- Add better prompt formatting and answer evaluation
