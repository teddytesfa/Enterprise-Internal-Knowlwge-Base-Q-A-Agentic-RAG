# Enterprise Internal Knowledge Base — Ingestion & Indexing (v1.0)

This starter layout implements a local Markdown ingestion + indexing pipeline adapted from the arXiv-paper-curator ingestion architecture.

Goals:
- Read markdown files from a content folder
- Parse frontmatter and markdown into plain text
- Chunk documents into overlapping segments
- Generate embeddings (pluggable: sentence-transformers if available, otherwise TF-IDF fallback)
- Index vectors for nearest-neighbour retrieval (simple sklearn-based indexer)
- Persist minimal metadata in SQLite

Quickstart:
1. Create and activate a Python environment (Python 3.10+ recommended)
2. Install deps:
   pip install -r requirements.txt
3. Put markdown files in ./content/ (e.g., content/example.md)
4. Run the ingest script:
   python -m src.main --content_dir content --output_dir data
5. Start the API:
   uvicorn src.api.app:app --reload --port 8000
6. Test retrieval:
   GET http://localhost:8000/api/v1/health
   POST http://localhost:8000/api/v1/retrieve JSON body: {"query":"your question","top_k":5}

Files of interest:
- src/main.py            — ingestion orchestrator (CLI)
- src/services/*         — modular ingestion components
- src/repositories/*     — SQLite persistence for documents and chunks
- src/services/indexer/* — simple sklearn indexer that saves pickles to disk
- src/api/*              — FastAPI wrapper for retrieval

Notes:
- The embedding service uses sentence-transformers if available. If not installed, a TF-IDF fallback is used.
- The indexer is file-backed and simple (sklearn.NearestNeighbors). Replace with FAISS or a vector DB later.
- For production, replace SQLite with PostgreSQL and add scheduling (Airflow) as needed.