from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from src.services.retrieval import Retriever
from src.dependencies import get_repo_and_indexer

router = APIRouter()

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrieveItem(BaseModel):
    score: float
    doc_id: int
    position: int
    title: str | None
    source: str | None
    chunk: str | None

class RetrieveResponse(BaseModel):
    results: List[RetrieveItem]

def get_retriever():
    repo, indexer, embed = get_repo_and_indexer()
    return Retriever(indexer=indexer, repo=repo, embed_service=embed)

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, retriever: Retriever = Depends(get_retriever)):
    if retriever.indexer._nn is None:
        raise HTTPException(status_code=503, detail="Index not available. Run ingestion first.")
    results = retriever.retrieve(req.query, top_k=req.top_k)
    items = [RetrieveItem(**r) for r in results]
    return RetrieveResponse(results=items)

@router.get("/health")
def health():
    return {"status": "ok"}
