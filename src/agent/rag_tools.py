from agent.tool import tool, Tool

def create_rag_tool(index) -> Tool:
    """
    Creates a Tool instance for RAG querying on the provided index.
    
    Args:
        index: The LlamaIndex vector index.
        
    Returns:
        Tool: The RAG query tool.
    """
    query_engine = index.as_query_engine()

    @tool
    def rag_query_engine(query: str):
        """
        Queries the internal knowledge base to answer questions about company policies, documents, etc.
        
        Args:
            query (str): The natural language query to search for.
            
        Returns:
            str: The response from the knowledge base.
        """
        response = query_engine.query(query)
        # We might want to format this later to include sources, 
        # but for now returning the string response is sufficient for the basic interaction.
        return str(response)

    return rag_query_engine
