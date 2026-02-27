import requests

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
CHAT_MODEL = "qwen2.5:7b"

def build_prompt(context: str, question: str) -> str:
    return f"""
You are a precise AI assistant.

Rules:
- Answer ONLY using the provided context.
- If information is missing, say:
"I don't have enough information in the provided context."
- Do not hallucinate.

Context:
{context}

Question:
{question}
"""

def chat(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=60
    )

    resp.raise_for_status()
    data = resp.json()

    return data["message"]["content"]