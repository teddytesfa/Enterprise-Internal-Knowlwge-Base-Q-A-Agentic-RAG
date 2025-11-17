# LlamaIndex core imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)

from llama_index.vector_stores.chroma import ChromaVectorStore

# ChromaDB for vector storage
import chromadb

from config.config import initialize_llamaindex_settings, VECTOR_DB_DIR

# Initialize settings first
initialize_llamaindex_settings()

def load_vector_index(VECTOR_DB_DIR: str=VECTOR_DB_DIR) -> VectorStoreIndex:
    """Load existing vector index from ChromaDB."""

    print("🔄 Loading existing vector index from Vector DB storage...")

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection_name = "internal_knowledge_base"

    # Load the collection
    try:
       chroma_collection = chroma_client.get_collection(name=collection_name)
       print(f"✅ Found existing collection: {collection_name}")
       print(f"   Total vectors: {chroma_collection.count()}")
    except Exception as e:
       print(f"❌ Error loading collection: {e}")
       print("   Please run ingestion pipeline first to create the vector index.")
       raise

    # Create ChromaVectorStore wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(VECTOR_DB_DIR))

    # Load the index
    try:
        index = load_index_from_storage(storage_context)   
        print("✅ Vector index loaded successfully!")
        return index
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        raise