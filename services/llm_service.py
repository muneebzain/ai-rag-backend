import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a helpful AI assistant.
Only answer using the provided context.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""


def chat(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]


def chat_stream(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))

                if "message" in data:
                    content = data["message"].get("content", "")
                    if content:
                        yield content