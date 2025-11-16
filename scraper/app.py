from fastapi import FastAPI
import requests
import trafilatura

app = FastAPI()

@app.get("/scrape")
def scrape(url: str):
    try:
        html = requests.get(url, timeout=10).text
        text = trafilatura.extract(html) or ""
        return {
            "url": url,
            "content": text
        }
    except Exception as e:
        return {"url": url, "content": "", "error": str(e)}
