import requests
from typing import List

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []

    for t in texts:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=20
        )
        resp.raise_for_status()
        data = resp.json()

        if "embedding" not in data:
            raise RuntimeError("Embedding missing from response")

        vectors.append(data["embedding"])

    return vectors