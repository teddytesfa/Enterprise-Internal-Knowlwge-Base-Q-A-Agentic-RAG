from src.config import settings
from src.repositories.document_repo import DocumentRepo
from src.services.indexer.simple_indexer import SimpleIndexer
from src.services.embeddings.factory import EmbeddingService
from pathlib import Path

# Try to import llama-index loader
try:
    from src.services.llama_index_service import load_index, LlamaIndexUnavailable, LlamaIndexWrapper  # type: ignore
    _llama_support = True
except Exception:
    load_index = None
    LlamaIndexUnavailable = Exception
    LlamaIndexWrapper = None
    _llama_support = False


def get_repo_and_indexer():
    output_dir = Path(settings.OUTPUT_DIR)
    db_url = f"sqlite:///{output_dir / settings.DB_PATH.name}"
    repo = DocumentRepo(db_url)

    # If llama-index persists an index, try to load it and return a wrapper exposing .query(...)
    if _llama_support:
        try:
            idx = load_index(settings.INDEX_DIR)
            return repo, idx, EmbeddingService(dim=settings.EMBEDDING_DIM)
        except Exception:
            # unable to load llama-index (missing files or import error) -> fall back
            pass

    # fallback: SimpleIndexer
    indexer = SimpleIndexer(str(settings.INDEX_DIR))
    try:
        indexer.load(prefix="v1")
    except Exception:
        # index files missing — indexer will be empty
        pass
    embed = EmbeddingService(dim=settings.EMBEDDING_DIM)
    return repo, indexer, embed
