
import os
from pathlib import Path
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import dotenv

# Load environment variables
dotenv.load_dotenv()

# --- PATHS ---
PROJECT_ROOT = Path("/Users/teddytesfa/projects/AI-data-science-and-ML/Enterprise Internal Knowlwge Base Q&A Agentic RAG")
SAMPLE_DATA_DIR = PROJECT_ROOT / "resources" / "sample-datasets"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

# --- API KEYS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- LLAMAINDEX SETTINGS ---
def initialize_llamaindex_settings():
    """Initializes global settings for LlamaIndex."""
    
    if not GOOGLE_API_KEY:
        print("⚠️  WARNING: GOOGLE_API_KEY not found in environment variables.")
        print("   Please set it using: export GOOGLE_API_KEY='your-api-key'")

    # Set up embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=str(PROJECT_ROOT / "models")
    )

    # Set up LLM
    llm = Gemini(
        model="models/gemini-2.5-flash",
        api_key=GOOGLE_API_KEY if GOOGLE_API_KEY else None
    )

    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    print("✅ LlamaIndex settings configured:")
    print(f"   - Embedding Model: {embed_model.model_name}")
    print(f"   - LLM: {llm.model}")
    print(f"   - Chunk Size: {Settings.chunk_size}")
    print(f"   - Chunk Overlap: {Settings.chunk_overlap}")

