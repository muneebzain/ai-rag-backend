import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel

from services.chunking import chunk_text
from services.embed_service import embed_texts
from services.vector_store import add_chunks, search
from services.llm_service import chat, build_prompt

app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory cache
ASK_CACHE = {}

# ----------------------------
# Request Models
# ----------------------------

class IngestRequest(BaseModel):
    doc_id: str
    text: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


# ----------------------------
# Utility Functions
# ----------------------------

def error_response(code: str, message: str):
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def root():
    return {"success": True, "message": "AI RAG Backend Running"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        chunks = chunk_text(req.text)
        vectors = embed_texts(chunks)
        add_chunks(req.doc_id, chunks, vectors)

        return {
            "success": True,
            "data": {
                "doc_id": req.doc_id,
                "chunks_added": len(chunks)
            }
        }

    except Exception as e:
        return error_response("INGEST_FAILED", str(e))


@app.post("/ask")
def ask(req: AskRequest):
    total_start = time.perf_counter()

    question = req.question.strip()
    if not question:
        return error_response("EMPTY_QUESTION", "Question cannot be empty")

    top_k = max(1, min(req.top_k, 10))

    question_norm = " ".join(question.lower().split())
    cache_key = f"{question_norm}|{top_k}"

    # ----------------------------
    # Cache
    # ----------------------------
    if cache_key in ASK_CACHE:
        logger.info("Cache hit")
        cached = ASK_CACHE[cache_key]
        total_ms = int((time.perf_counter() - total_start) * 1000)

        return {
            "success": True,
            "data": cached,
            "meta": {
                "performance": {
                    "cached": True,
                    "total_ms": total_ms
                }
            }
        }

    logger.info(f"Question: {question}")

    # ----------------------------
    # Embed
    # ----------------------------
    t1 = time.perf_counter()
    q_vec = embed_texts([question])[0]
    embed_ms = int((time.perf_counter() - t1) * 1000)

    # ----------------------------
    # Search
    # ----------------------------
    t2 = time.perf_counter()
    results = search(q_vec, top_k=top_k)
    search_ms = int((time.perf_counter() - t2) * 1000)

    docs = results.get("documents", [[]])[0]
    scores = results.get("distances", [[]])[0]

    # ----------------------------
    # Hybrid Keyword Boost
    # ----------------------------
    question_words = set(question.lower().split())
    scored_docs = []

    for i, d in enumerate(docs):
        doc_words = set(d.lower().split())
        keyword_overlap = len(question_words & doc_words)
        distance = scores[i] if i < len(scores) else 1.0
        scored_docs.append((keyword_overlap - distance, d))

    scored_docs.sort(reverse=True)
    docs = [d for _, d in scored_docs]

    # ----------------------------
    # Limit Context Length
    # ----------------------------
    MAX_CONTEXT_CHARS = 3000
    context_text = ""

    for d in docs:
        if len(context_text) + len(d) > MAX_CONTEXT_CHARS:
            break
        context_text += d + "\n\n"

    # ----------------------------
    # Prompt + LLM
    # ----------------------------
    prompt = build_prompt(context_text, question)

    t3 = time.perf_counter()
    answer = chat(prompt)
    llm_ms = int((time.perf_counter() - t3) * 1000)

    # ----------------------------
    # Confidence (Normalized)
    # ----------------------------
    if scores:
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score > 0:
            normalized_scores = [
                (max_score - s) / (max_score - min_score)
                for s in scores
            ]
            confidence = sum(normalized_scores) / len(normalized_scores)
        else:
            confidence = 0.5
    else:
        confidence = 0.0

    # ----------------------------
    # Confidence Threshold Refusal
    # ----------------------------
    CONFIDENCE_THRESHOLD = 0.35

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info("Low confidence — refusing answer")
        return {
            "success": True,
            "data": {
                "answer": "I don't have enough reliable information in the provided context.",
                "sources": [],
                "confidence": round(confidence, 2),
                "refused": True
            },
            "meta": {
                "performance": {
                    "embed_ms": embed_ms,
                    "search_ms": search_ms,
                    "llm_ms": llm_ms,
                    "total_ms": int((time.perf_counter() - total_start) * 1000)
                }
            }
        }

    # ----------------------------
    # Context Coverage Check (Grounding)
    # ----------------------------
    answer_words = set(answer.lower().split())
    context_words = set(context_text.lower().split())

    overlap = len(answer_words & context_words)
    coverage_ratio = overlap / max(1, len(answer_words))

    if coverage_ratio < 0.2:
        logger.info("Low context coverage — refusing answer")
        return {
            "success": True,
            "data": {
                "answer": "The generated answer appears insufficiently grounded in the retrieved context.",
                "sources": [],
                "confidence": round(confidence, 2),
                "refused": True
            },
            "meta": {
                "performance": {
                    "embed_ms": embed_ms,
                    "search_ms": search_ms,
                    "llm_ms": llm_ms,
                    "total_ms": int((time.perf_counter() - total_start) * 1000)
                }
            }
        }

    total_ms = int((time.perf_counter() - total_start) * 1000)

    response_data = {
        "answer": answer,
        "sources": docs,
        "confidence": round(confidence, 2),
        "refused": False
    }

    ASK_CACHE[cache_key] = response_data

    logger.info(f"Total time: {total_ms}ms")

    return {
        "success": True,
        "data": response_data,
        "meta": {
            "performance": {
                "embed_ms": embed_ms,
                "search_ms": search_ms,
                "llm_ms": llm_ms,
                "total_ms": total_ms
            }
        }
    }