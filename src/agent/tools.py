"""Tool definitions for ReAct agent."""

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex

def create_query_tool(index: VectorStoreIndex) -> QueryEngineTool:
    """Create QueryEngineTool from vector index."""

    print("🔧 Setting up QueryEngineTool...")

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=False
    )
    
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_knowledge_base",
            description=(
                "Search the internal company knowledge base for information. "
                "Use this tool when you need to find specific information from "
                "documents like HR policies, technical guides, or meeting notes."
            )
        )
    )
    return tool