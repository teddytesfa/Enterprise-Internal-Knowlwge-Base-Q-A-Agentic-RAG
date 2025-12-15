import os
from pathlib import Path
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.core.llms import LLM
import dotenv

# --- Project root (relative) ---
# Compute the project root relative to this file (src/config/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from the project-root `.env` explicitly so the
# settings work regardless of the current working directory.
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    dotenv.load_dotenv(dotenv_path=dotenv_path)
else:
    # fallback to default behavior (load from environment)
    dotenv.load_dotenv()

# --- PATHS ---
SAMPLE_DATA_DIR = PROJECT_ROOT / "resources" / "sample-datasets"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

# --- API KEYS ---
# Read GOOGLE_API_KEY from environment (no hard-coded secret in code)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- LLAMAINDEX SETTINGS ---
def get_agent_llm() -> LLM:
    """
    Factory to return the configured LLM based on environment variables.
    Defaults to Gemini if LLM_PROVIDER is not set or set to 'gemini'.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        if not GROQ_API_KEY:
            # Fallback or error handling
            print("⚠️  WARNING: GROQ_API_KEY not found. Attempting to use default or failing.")
        
        return Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    
    elif provider == "gemini":
        return Gemini(
             model="models/gemini-2.5-flash",
             api_key=GOOGLE_API_KEY
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


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
    llm = get_agent_llm()

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
