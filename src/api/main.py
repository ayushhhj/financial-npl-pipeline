import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Financial NLP Pipeline API",
    description="Semantic search and knowledge graph over SEC 10-K filings",
    version="1.0.0",
)

EMBEDDINGS_DIR = Path("data/embeddings")

model = SentenceTransformer("all-MiniLM-L6-v2")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password"),
    ),
)

index = faiss.read_index(str(EMBEDDINGS_DIR / "index.faiss"))
with open(EMBEDDINGS_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
with open(EMBEDDINGS_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    score: float
    ticker: str
    date: str
    text: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Financial NLP Pipeline API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": len(chunks),
        "index_size": index.ntotal,
    }


@app.post("/search", response_model=list[SearchResult])
def semantic_search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = model.encode([request.query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, request.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append(SearchResult(
            score=round(float(score), 4),
            ticker=metadata[idx]["ticker"],
            date=metadata[idx]["date"],
            text=chunks[idx][:500],
        ))
    return results


@app.get("/entities/{ticker}")
def get_entities(ticker: str):
    ticker = ticker.upper()
    with driver.session() as session:
        company = session.run(
            "MATCH (c:Company {id: $ticker}) RETURN c",
            ticker=ticker,
        ).single()

        if not company:
            raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

        persons = session.run(
            """
            MATCH (p:Person)-[r:ASSOCIATED_WITH]->(c:Company {id: $ticker})
            RETURN p.name as name, r.count as count
            ORDER BY r.count DESC LIMIT 20
            """,
            ticker=ticker,
        ).data()

        locations = session.run(
            """
            MATCH (c:Company {id: $ticker})-[:FILED]->(f:Filing)-[r:MENTIONS_LOCATION]->(l:Location)
            RETURN l.name as name, sum(r.count) as count
            ORDER BY count DESC LIMIT 20
            """,
            ticker=ticker,
        ).data()

        filings = session.run(
            """
            MATCH (c:Company {id: $ticker})-[:FILED]->(f:Filing)
            RETURN f.accession as accession, f.date as date
            ORDER BY f.date DESC
            """,
            ticker=ticker,
        ).data()

    return {
        "ticker": ticker,
        "persons": persons,
        "locations": locations,
        "filings": filings,
    }


@app.get("/graph/{ticker}/related")
def get_related_companies(ticker: str):
    ticker = ticker.upper()
    with driver.session() as session:
        company = session.run(
            "MATCH (c:Company {id: $ticker}) RETURN c",
            ticker=ticker,
        ).single()

        if not company:
            raise HTTPException(status_code=404, detail=f"Company {ticker} not found")

        co_mentions = session.run(
            """
            MATCH (c1:Company {id: $ticker})-[r:CO_MENTIONED_WITH]->(c2:Company)
            RETURN c2.ticker as ticker, c2.name as name, r.count as count
            ORDER BY r.count DESC
            """,
            ticker=ticker,
        ).data()

        shared_persons = session.run(
            """
            MATCH (p:Person)-[:ASSOCIATED_WITH]->(c1:Company {id: $ticker})
            MATCH (p)-[:ASSOCIATED_WITH]->(c2:Company)
            WHERE c2.id <> $ticker
            RETURN c2.ticker as ticker, c2.name as name, collect(p.name) as shared_persons
            ORDER BY size(collect(p.name)) DESC LIMIT 10
            """,
            ticker=ticker,
        ).data()

    return {
        "ticker": ticker,
        "co_mentions": co_mentions,
        "shared_persons": shared_persons,
    }


@app.get("/companies")
def list_companies():
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Company)
            OPTIONAL MATCH (c)-[:FILED]->(f:Filing)
            RETURN c.ticker as ticker, c.name as name, count(f) as filing_count
            ORDER BY c.ticker
            """
        ).data()
    return result