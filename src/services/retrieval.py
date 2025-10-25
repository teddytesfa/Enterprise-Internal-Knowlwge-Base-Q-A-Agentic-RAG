from typing import List, Dict, Any
import numpy as np
from src.services.embeddings.factory import EmbeddingService
from src.services.indexer.simple_indexer import SimpleIndexer
from src.repositories.document_repo import DocumentRepo

class Retriever:
    def __init__(self, indexer: SimpleIndexer, repo: DocumentRepo, embed_service: EmbeddingService):
        self.indexer = indexer
        self.repo = repo
        self.embed_service = embed_service

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # embed query
        q_vec = self.embed_service.embed_documents([query])
        # query index
        results = self.indexer.query(q_vec[0], top_k=top_k)
        out = []
        for meta, score in results:
            doc_id = meta.get("doc_id")
            pos = meta.get("position")
            chunk_text = self.repo.get_chunk_text(doc_id, pos)
            doc = self.repo.get_document(doc_id)
            out.append({
                "score": score,
                "doc_id": doc_id,
                "position": pos,
                "chunk": chunk_text,
                "title": doc.title if doc else None,
                "source": meta.get("source"),
            })
        return out
