from fastapi import FastAPI
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

app = FastAPI()

SEARX_URL = "http://searxng:8080/search"
SCRAPER_URL = "http://scraper:8090/scrape"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(host="qdrant", port=6333)

# Create collection
client.recreate_collection(
    collection_name="web_memory",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

@app.get("/search")
def search(query: str):
    # Step 1: run search in SearxNG
    params = {"q": query, "format": "json"}
    results = requests.get(SEARX_URL, params=params, timeout=10).json()

    urls = [r["url"] for r in results.get("results", [])[:5]]

    combined_text = ""

    # Step 2: scrape each result
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
            pass

    return {
        "query": query,
        "sources": urls,
        "context": combined_text[:15000]  # limit for LLM context
    }

@app.get("/memory")
def memory(query: str):
    embedding = model.encode(query).tolist()

    results = client.search(
        collection_name="web_memory",
        query_vector=embedding,
        limit=5
    )

    return {
        "query": query,
        "matches": [{
            "url": r.payload["url"],
            "text": r.payload["text"][:2000]
        } for r in results]
    }
