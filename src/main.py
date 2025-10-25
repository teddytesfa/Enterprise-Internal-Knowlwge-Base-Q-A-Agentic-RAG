import argparse
from pathlib import Path
from src.config import settings
from src.services.local_reader import list_markdown_files, read_markdown
from src.services.markdown_parser import markdown_to_text
from src.services.text_chunker import TextChunker
from src.services.embeddings.factory import EmbeddingService
from src.services.indexer.simple_indexer import SimpleIndexer
from src.repositories.document_repo import DocumentRepo
import numpy as np
import json

def run_ingest(content_dir: Path, output_dir: Path):
    content_dir = content_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup DB
    db_url = f"sqlite:///{output_dir / settings.DB_PATH.name}"
    repo = DocumentRepo(db_url)

    # Setup chunker & embedding service
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    embed_service = EmbeddingService(dim=settings.EMBEDDING_DIM)

    # Gather documents
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

    # Build embeddings
    texts = [d["text"] for d in docs]
    if len(texts) == 0:
        print("No documents found. Exiting.")
        return
    vectors = embed_service.embed_documents(texts)
    # Simple indexer
    indexer = SimpleIndexer(str(output_dir / "index"))
    metadata = [{"doc_id": d["doc_id"], "position": d["position"], "source": d["source"], "title": d["title"]} for d in docs]
    indexer.build(vectors, metadata)
    indexer.save(prefix="v1")
    # also persist a small manifest
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"documents": len(docs)}, fh, indent=2)
    print(f"Ingested {len(docs)} chunks. DB: {db_url}, index: {output_dir / 'index'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", default=str(settings.CONTENT_DIR))
    parser.add_argument("--output_dir", default=str(settings.OUTPUT_DIR))
    args = parser.parse_args()
    run_ingest(Path(args.content_dir), Path(args.output_dir))
