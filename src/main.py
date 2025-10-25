"""
Ingestion entrypoint.

Now tries to use llama-index if available. If llama-index is installed and works,
we will build and persist a llamaIndex index from the content directory (fast path).
If llama-index is not available, we fall back to the previous pipeline (chunk -> embeddings -> SimpleIndexer).
"""
import argparse
from pathlib import Path
from src.config import settings
from src.services.local_reader import list_markdown_files, read_markdown
from src.services.markdown_parser import markdown_to_text
from src.services.text_chunker import TextChunker
from src.services.embeddings.factory import EmbeddingService
from src.services.indexer.simple_indexer import SimpleIndexer
from src.repositories.document_repo import DocumentRepo
import json
import sys

# Try to import llama-index helper (may raise or be unavailable)
try:
    from src.services.llama_index_service import build_index_from_dir, LlamaIndexUnavailable
    _llama_ok = True
except Exception:
    build_index_from_dir = None
    LlamaIndexUnavailable = Exception
    _llama_ok = False

def run_ingest(content_dir: Path, output_dir: Path):
    content_dir = content_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup DB
    db_url = f"sqlite:///{output_dir / settings.DB_PATH.name}"
    repo = DocumentRepo(db_url)

    # If llama-index available, prefer it (reads raw files directly and builds index)
    if _llama_ok and build_index_from_dir is not None:
        try:
            print("Building llama-index from content directory...")
            idx = build_index_from_dir(content_dir, settings.INDEX_DIR)
            print(f"llama-index built and persisted at {settings.INDEX_DIR}")
            # We still record minimal metadata in DB (document entries) for compatibility
            count = 0
            for md_path in list_markdown_files(content_dir):
                metadata, md_text = read_markdown(md_path)
                plain = markdown_to_text(md_text)
                title = metadata.get("title") if isinstance(metadata, dict) else md_path.stem
                doc = repo.upsert_document(str(md_path), title, metadata or {})
                # naive: chunking still persisted for chunk-level storage used by Retriever fallback
                chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
                chunks = chunker.chunk_text(plain)
                repo.add_chunks(doc.id, chunks)
                count += len(chunks)
            # Save manifest
            with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump({"chunks": count}, fh, indent=2)
            print(f"Ingested {count} chunks (metadata persisted).")
            return
        except LlamaIndexUnavailable as e:
            print("llama-index not available, falling back to simple pipeline:", e, file=sys.stderr)
        except Exception as e:
            print("llama-index build failed, falling back to simple pipeline:", e, file=sys.stderr)

    # Fallback pipeline (previous behavior): chunk -> embed -> simple indexer
    print("Using fallback ingestion pipeline (chunk -> embed -> SimpleIndexer)")
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    embed_service = EmbeddingService(dim=settings.EMBEDDING_DIM)

    docs = []
    for md_path in list_markdown_files(content_dir):
        metadata, md_text = read_markdown(md_path)
        plain = markdown_to_text(md_text)
        title = metadata.get("title") if isinstance(metadata, dict) else md_path.stem
        # idempotent upsert
        doc = repo.upsert_document(str(md_path), title, metadata or {})
        # chunk
        chunks = chunker.chunk_text(plain)
        repo.add_chunks(doc.id, chunks)
        for i, c in enumerate(chunks):
            docs.append({
                "doc_id": doc.id,
                "source": str(md_path),
                "position": i,
                "text": c,
                "title": title,
            })

    texts = [d["text"] for d in docs]
    if len(texts) == 0:
        print("No documents found. Exiting.")
        return
    vectors = embed_service.embed_documents(texts)
    indexer = SimpleIndexer(str(output_dir / "index"))
    metadata = [{"doc_id": d["doc_id"], "position": d["position"], "source": d["source"], "title": d["title"]} for d in docs]
    indexer.build(vectors, metadata)
    indexer.save(prefix="v1")
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"documents": len(docs)}, fh, indent=2)
    print(f"Ingested {len(docs)} chunks. DB: {db_url}, index: {output_dir / 'index'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default=str(settings.CONTENT_DIR))
    parser.add_argument("--output_dir", default=str(settings.OUTPUT_DIR))
    args = parser.parse_args()
    run_ingest(Path(args.content_dir), Path(args.output_dir))