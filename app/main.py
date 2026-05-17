from fastapi import FastAPI

from app.schemas import QueryRequest, QueryResponse

app = FastAPI(title="RAG Engine API")


@app.get("/")
async def root():
    return {"message": "RAG Engine API is running"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    return QueryResponse(answer="This is a placeholder answer.", sources=[])
