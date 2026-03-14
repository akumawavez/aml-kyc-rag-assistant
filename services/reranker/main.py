"""
Reranker service: proxies requests to Cohere Rerank API.
POST /rerank with JSON body: { "query": "...", "documents": ["doc1", "doc2", ...], "top_n": 3 }
Returns: { "results": [ { "index": 0, "relevance_score": 0.99 }, ... ] }
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Cohere Reranker Proxy", version="1.0")


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int = 10


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="COHERE_API_KEY not set; reranker service unavailable",
        )
    try:
        import cohere

        client = cohere.ClientV2(api_key=api_key)
        response = client.rerank(
            model="rerank-english-v3.0",
            query=req.query,
            documents=req.documents,
            top_n=min(req.top_n, len(req.documents)),
        )
        results = [
            RerankResult(index=r.index, relevance_score=r.relevance_score)
            for r in response.results
        ]
        return RerankResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cohere_configured": bool(os.environ.get("COHERE_API_KEY"))}
