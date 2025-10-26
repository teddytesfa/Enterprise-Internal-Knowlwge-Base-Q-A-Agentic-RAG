"""
llama_index wrapper service.

This module provides a small wrapper around llama-index to:
- build/persist an index from a content directory (markdown files)
- load a persisted index
- query the index returning a simple standardized result list

If llama-index is not installed or an error occurs, the functions will raise ImportError
so the caller can fallback to the simple indexer.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

class LlamaIndexUnavailable(Exception):
    pass

try:
    # llama-index public API varies by version; attempt to import common utilities.
    from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage, GPTVectorStoreIndex
    from llama_index.embeddings import HuggingFaceEmbedding
    _llama_import_error = None
except Exception as e:
    SimpleDirectoryReader = None  # type: ignore
    StorageContext = None  # type: ignore
    load_index_from_storage = None  # type: ignore
    GPTVectorStoreIndex = None  # type: ignore
    HuggingFaceEmbedding = None  # type: ignore
    _llama_import_error = e

class LlamaIndexWrapper:
    def __init__(self, index, storage_context: Optional[object] = None):
        self._index = index
        self._storage_context = storage_context

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the llama-index and return a list of items with fields:
        - 'score' (float)
        - 'doc_id' (optional)
        - 'source' (source file path if available)
        - 'chunk' (text snippet)
        - 'extra' (raw node or metadata)
        """
        # Use the query engine if available
        try:
            # Newer llama-index versions expose as_query_engine(...)
            if hasattr(self._index, "as_query_engine"):
                qeng = self._index.as_query_engine(similarity_top_k=top_k)
                response = qeng.query(query_text)
                nodes = []
                if hasattr(response, "source_nodes") and response.source_nodes:
                    nodes = response.source_nodes
                elif hasattr(response, "response_nodes") and response.response_nodes:
                    nodes = response.response_nodes
                results = []
                for n in nodes[:top_k]:
                    # node text extraction
                    text = getattr(n, "get_text", lambda: str(n))()
                    meta = getattr(n, "source", None) or getattr(n, "node_info", None) or {}
                    results.append({
                        "score": float(getattr(n, "score", 0.0) or 0.0),
                        "doc_id": meta.get("doc_id") if isinstance(meta, dict) else None,
                        "source": meta.get("source") if isinstance(meta, dict) else None,
                        "chunk": text,
                        "extra": meta,
                    })
                return results
            else:
                # fallback older API
                resp = self._index.query(query_text, similarity_top_k=top_k)
                nodes = []
                if hasattr(resp, "source_nodes") and resp.source_nodes:
                    nodes = resp.source_nodes
                elif hasattr(resp, "response_nodes") and resp.response_nodes:
                    nodes = resp.response_nodes
                results = []
                for n in nodes[:top_k]:
                    text = getattr(n, "get_text", lambda: str(n))()
                    meta = getattr(n, "source", None) or getattr(n, "node_info", None) or {}
                    results.append({
                        "score": float(getattr(n, "score", 0.0) or 0.0),
                        "doc_id": meta.get("doc_id") if isinstance(meta, dict) else None,
                        "source": meta.get("source") if isinstance(meta, dict) else None,
                        "chunk": text,
                        "extra": meta,
                    })
                return results
        except Exception:
            # Best-effort: return empty list on error
            return []

def build_index_from_dir(content_dir: Path, persist_dir: Path, hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds and persists a llama-index vector index from the given content directory.
    Expects content_dir to include markdown files. Returns a LlamaIndexWrapper.
    Raises LlamaIndexUnavailable if llama-index is missing.
    """
    if SimpleDirectoryReader is None:
        raise LlamaIndexUnavailable(f"llama-index import failed: {_llama_import_error}")

    # Load documents from directory (SimpleDirectoryReader reads markdown, txt, etc.)
    docs = SimpleDirectoryReader(str(content_dir)).load_data()

    # Create embedding model for llama-index
    embed = HuggingFaceEmbedding(model_name=hf_model_name)

    # Create a storage context to persist index
    persist_dir.mkdir(parents=True, exist_ok=True)
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))

    # Build the index using the embedding model
    index = GPTVectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed)

    # Persist the index to disk
    index.storage_context.persist(persist_dir=str(persist_dir))

    return LlamaIndexWrapper(index, storage_context)

def load_index(persist_dir: Path) -> LlamaIndexWrapper:
    """
    Loads a persisted llama-index from disk. Raises LlamaIndexUnavailable if llama-index not installed.
    """
    if load_index_from_storage is None or StorageContext is None:
        raise LlamaIndexUnavailable(f"llama-index import failed: {_llama_import_error}")
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    return LlamaIndexWrapper(index, storage_context)