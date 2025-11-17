from config.config import initialize_llamaindex_settings, SAMPLE_DATA_DIR, VECTOR_DB_DIR
from connector import DocumentConnector
from indexer import chunk_documents, build_index
from IPython.display import display, Markdown
import time

def main():
    """Main function to run the document ingestion and indexing pipeline."""
    
    # --- 1. Initialize Settings ---
    initialize_llamaindex_settings()

    # --- 2. Load Documents ---
    print("\n🔄 Loading documents...")
    connector = DocumentConnector(SAMPLE_DATA_DIR)
    connector.display_summary()
    documents = connector.load_documents()
    if not documents:
        print("⚠️  No documents loaded. Exiting.")
        return
    print(f"✅ Successfully loaded {len(documents)} document(s)")

    # --- 3. Chunk Documents ---
    print("\n🔄 Chunking documents...")
    nodes = chunk_documents(documents)

    # --- 4. Build Index ---
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    index = build_index(nodes, VECTOR_DB_DIR)

    # --- 5. Test Retrieval ---
    print("\n🚀 Executing test queries...")
    test_retrieval(index)

def test_retrieval(index):
    """Tests the retrieval quality of the index."""
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        streaming=False
    )

    test_queries = [
        {
            "category": "Policy Lookup (Direct Factual Retrieval)",
            "query": "What is our parental leave policy?"
        },
        {
            "category": "Developer Environment Setup (Step-by-step)",
            "query": "How do I set up the local dev environment?"
        },
        {
            "category": "Edge Case: Out-of-Domain",
            "query": "What's the weather today in London?"
        }
    ]

    for test in test_queries:
        query_str = test["query"]
        category = test["category"]

        print(f"\n{'='*80}\n## ❓ Query: '{query_str}'\n**Category**: {category}\n{'='*80}")

        start_time = time.time()
        response = query_engine.query(query_str)
        response_time = time.time() - start_time

        print(f"### 💬 Synthesized Answer (in {response_time:.2f}s)")
        print(f"> {response.response}")

        print("### 📚 Retrieved Chunks (Sources)")
        if response.source_nodes:
            for i, node in enumerate(response.source_nodes):
                source_file = node.metadata.get('file_name', 'Unknown')
                preview = node.text.replace('\n', ' ').strip()[:100] + "..."
                relevance_feedback = "✅ Relevant" if node.score > 0.7 else ("🤔 Potentially Relevant" if node.score > 0.6 else "⚠️ Low Relevance")
                print(f"**Chunk {i+1} | Score: {node.score:.3f} ({relevance_feedback}) | Source: {source_file}**")
                print(f"```\n{preview}\n```")
        else:
            print("*No relevant chunks found.*")

if __name__ == "__main__":
    main()
