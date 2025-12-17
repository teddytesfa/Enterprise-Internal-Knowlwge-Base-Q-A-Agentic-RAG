
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import VECTOR_DB_DIR
from storage.storage import load_vector_index
from agent.react_agent import ReactAgent
from agent.rag_tools import create_rag_tool

app = FastAPI()

index = load_vector_index(VECTOR_DB_DIR)

# Initialize ReAct agent with the loaded index via tool
rag_tool = create_rag_tool(index)
react_agent = ReactAgent(tools=[rag_tool])

class Query(BaseModel):
    question: str

@app.post("/query")
async def run_query(query: Query):
    """Execute query through ReAct agent."""
    # The new ReactAgent.run returns a single string response.
    # Future improvements can parse this back into structured output if needed.
    result_str = react_agent.run(query.question)
    
    return result_str

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
