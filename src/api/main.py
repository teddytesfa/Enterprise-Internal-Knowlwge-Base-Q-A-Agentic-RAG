
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import VECTOR_DB_DIR
from storage.storage import load_vector_index
from agent.react_agent import ReActRAGAgent

app = FastAPI()

index = load_vector_index(VECTOR_DB_DIR)

# Initialize ReAct agent with the loaded index
react_agent = ReActRAGAgent(index, verbose=True)

class Query(BaseModel):
    question: str

@app.post("/query")
async def run_query(query: Query):
    """Execute query through ReAct agent."""
    result = await react_agent.query(query.question)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "reasoning_steps": result["reasoning_steps"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
