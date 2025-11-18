from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI(
    title="RAG Web Tool",
    description="OpenAPI tool for web search, scrape, and memory retrieval",
    version="1.0.0"
)

# Allow Open WebUI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# External services
SEARX_URL = "http://searxng:8080/search"
SCRAPER_URL = "http://scraper:8090/scrape"

# Models
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="qdrant", port=6333)

# Create collection
client.recreate_collection(
    collection_name="web_memory",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Pydantic input models for OpenAPI
class QueryRequest(BaseModel):
    query: str

# --- Tool endpoints --- #

@app.post("/tool/search")
def tool_search(request: QueryRequest):
    query = request.query

    # 1. Search SearxNG
    params = {"q": query, "format": "json"}
    try:
        results = requests.get(SEARX_URL, params=params, timeout=10).json()
    except:
        results = {"results": []}

    urls = [r["url"] for r in results.get("results", [])[:5]]

    combined_text = ""

    # 2. Scrape each result
    for url in urls:
        try:
            scraped = requests.get(SCRAPER_URL, params={"url": url}).json()
            text = scraped.get("content", "")
            if text:
                combined_text += f"\n\nSOURCE: {url}\n{text}\n"

                # Save to vector DB
                embedding = model.encode(text).tolist()
                client.upsert(
                    collection_name="web_memory",
                    points=[{
                        "id": url,
                        "vector": embedding,
                        "payload": {"url": url, "text": text}
                    }]
                )
        except:
            continue

    # Return OpenAPI tool-compatible output
    return {
        "name": "web_search",
        "description": "Performs web search and scrapes results for RAG context",
        "query": query,
        "sources": urls,
        "context": combined_text[:15000]
    }


@app.post("/tool/memory")
def tool_memory(request: QueryRequest):
    query = request.query
    embedding = model.encode(query).tolist()

    results = client.search(
        collection_name="web_memory",
        query_vector=embedding,
        limit=5
    )

    return {
        "name": "memory_lookup",
        "description": "Retrieves related documents from local memory",
        "query": query,
        "matches": [{
            "url": r.payload["url"],
            "text": r.payload["text"][:2000]
        } for r in results]
    }
