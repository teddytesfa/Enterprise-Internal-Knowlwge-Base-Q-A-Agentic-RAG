from pathlib import Path
import os

class Settings:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CONTENT_DIR = Path(os.getenv("CONTENT_DIR", PROJECT_ROOT / "content"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "data"))
    DB_PATH = OUTPUT_DIR / os.getenv("DB_NAME", "documents.db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))        # words
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))   # words
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384)) # default target dim
    INDEX_DIR = OUTPUT_DIR / "index"

settings = Settings()
