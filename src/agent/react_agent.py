"""ReAct agent initialization and management."""

from llama_index.core.agent import ReActAgent, AgentWorkflow
from llama_index.core import Settings
from config.config import initialize_llamaindex_settings
from typing import List
from .tools import create_query_tool
from .prompts import REACT_SYSTEM_PROMPT

class ReActRAGAgent:
    """ReAct agent for internal knowledge base Q&A."""
    
    def __init__(self, vector_index, verbose: bool = True):
        # Initialize settings first
        initialize_llamaindex_settings()

        self.index = vector_index
        self.verbose = verbose
        self.agent = self._initialize_agent()
    
    def _initialize_agent(self) -> ReActAgent:
        """Initialize ReAct agent with tools and prompts."""
        query_tool = create_query_tool(self.index)
        
        agent = ReActAgent(
            tools=[query_tool],
            llm=Settings.llm,
            verbose=self.verbose,
            max_iterations=10,
            system_prompt=REACT_SYSTEM_PROMPT
        )
        return agent
    
    async def query(self, question: str) -> dict:
        """Execute a query through the ReAct agent."""
        workflow = AgentWorkflow(
                    agents=[self.agent],
                    root_agent=self.agent.name,
                   )

        # Run the workflow
        handler = workflow.run(user_msg=question)
        response = await handler

        
        return {
            "answer": str(response),
            "sources": self._extract_sources(response),
            "reasoning_steps": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
        }
    
    def _extract_sources(self, response) -> List[str]:
        """Extract source document names from response."""
        # Implementation to parse citations
        pass