from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import time
from pathlib import Path
from typing import List
from llama_index.core.schema import Document

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunks the documents using SentenceSplitter."""
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separator=" ",
        paragraph_separator="\n\n",
    )
    print("✅ Node Parser (Chunker) Initialized")
    print(f"   - Type: SentenceSplitter")
    print(f"   - Chunk Size: 512 tokens")
    print(f"   - Chunk Overlap: 50 tokens")

    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    print(f"\n✅ Successfully created {len(nodes)} nodes from {len(documents)} document(s)")
    return nodes

def build_index(nodes: List[Document], vector_db_dir: Path) -> VectorStoreIndex:
    """Builds the vector index and stores it in ChromaDB."""
    collection_name = "internal_knowledge_base"
    
    print("\n🔄 Setting up ChromaDB vector store...")
    chroma_client = chromadb.PersistentClient(path=str(vector_db_dir))
    
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"   Deleted existing collection: {collection_name}")
    except:
        pass

    chroma_collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Internal company knowledge base for RAG"}
    )
    print(f"✅ ChromaDB collection created: {collection_name}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("\n🔄 Building vector index...")
    start_time = time.time()
    
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Vector index built successfully! Time taken: {elapsed_time:.2f} seconds")
    
    return index
