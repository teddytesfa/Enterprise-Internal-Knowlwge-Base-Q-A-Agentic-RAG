from fastapi import FastAPI
from src.api.routers import retrieve

app = FastAPI(title="Enterprise KB Retrieval (v1.0)")

app.include_router(retrieve.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"service": "Enterprise KB Retrieval", "version": "1.0"}
