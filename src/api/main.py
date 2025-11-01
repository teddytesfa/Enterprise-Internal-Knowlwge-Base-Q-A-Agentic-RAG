
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.config import initialize_llamaindex_settings, SAMPLE_DATA_DIR, VECTOR_DB_DIR
from ingestion.connector import DocumentConnector
from ingestion.indexer import chunk_documents, build_index

app = FastAPI()

# --- 1. Initialize Settings ---
initialize_llamaindex_settings()

# --- 2. Load Documents ---
print("\n🔄 Loading documents...")
connector = DocumentConnector(SAMPLE_DATA_DIR)
connector.display_summary()
documents = connector.load_documents()
if not documents:
    print("⚠️  No documents loaded. Exiting.")
    sys.exit(1)
print(f"✅ Successfully loaded {len(documents)} document(s)")

# --- 3. Chunk Documents ---
print("\n🔄 Chunking documents...")
nodes = chunk_documents(documents)

# --- 4. Build Index ---
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
index = build_index(nodes, VECTOR_DB_DIR)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",
    streaming=False
)

class Query(BaseModel):
    question: str

@app.post("/query")
async def run_query(query: Query):
    """Runs a query against the index."""
    response = query_engine.query(query.question)
    return {"response": response.response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
