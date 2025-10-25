from src.config import settings
from src.repositories.document_repo import DocumentRepo
from src.services.indexer.simple_indexer import SimpleIndexer
from src.services.embeddings.factory import EmbeddingService
from pathlib import Path

def get_repo_and_indexer():
    output_dir = Path(settings.OUTPUT_DIR)
    db_url = f"sqlite:///{output_dir / settings.DB_PATH.name}"
    repo = DocumentRepo(db_url)
    indexer = SimpleIndexer(str(settings.INDEX_DIR))
    # Try loading existing index (prefix v1)
    try:
        indexer.load(prefix="v1")
    except Exception:
        # index files missing — leave indexer empty
        pass
    embed = EmbeddingService(dim=settings.EMBEDDING_DIM)
    return repo, indexer, embed
