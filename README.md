# AI RAG Backend

Production-ready Retrieval-Augmented Generation (RAG) backend built with FastAPI.

## Features

- Document ingestion with chunking + overlap
- Embedding generation
- Vector similarity search
- Hybrid keyword boosting
- Context length limiting
- Confidence scoring
- Semantic grounding check
- Hallucination refusal logic
- In-memory caching
- Performance metrics tracking

## Tech Stack

- Python
- FastAPI
- Local Embeddings
- Vector Search
- Custom Grounding Layer

## Endpoints

POST /ingest  
POST /ask  

## Example Response

{
  "answer": "...",
  "confidence": 0.82,
  "grounding_score": 0.75,
  "refused": false
}

## Architecture

User Question  
→ Embed Query  
→ Vector Search  
→ Hybrid Ranking  
→ Context Assembly  
→ LLM  
→ Confidence + Grounding Evaluation  
→ Final Response

## Run Locally
Backend runs on:
http://127.0.0.1:8000