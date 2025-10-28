# app.py
import os
import time
import json
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# ---- Load knowledge base (pure Python, no Hyperon) ----
# Must be in the same folder:
#   knowledge_base.py  -> defines KNOWLEDGE_BASE: List[dict]
from knowledge_base import KNOWLEDGE_BASE

# ---- Environment & OpenAI ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var before starting the server")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- FastAPI app ----
app = FastAPI(
    title="Mental Health Chatbot API",
    description="RAG over knowledge_base.py with OpenAI embeddings + chat",
    version="1.0.0",
)

# CORS (adjust origins as needed for Agentverse / ASI:One)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Globals initialized at startup ----
EMBEDDING_DIM: int = 1536
EMB_MATRIX: Optional[np.ndarray] = None  # normalized embeddings [N, d]
KB_ITEMS: List[Dict[str, Any]] = []      # mirrors KNOWLEDGE_BASE (filtered if needed)


# ========= Models =========

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(3, ge=1, le=10, description="How many chunks to retrieve")
    threshold: float = Field(0.0, ge=-1.0, le=1.0, description="Cosine min score")
    embed_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    chat_model: str = Field("gpt-4o-mini", description="OpenAI chat model")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(800, ge=64, le=2000)
    safety_note: bool = Field(True, description="Append crisis/safety footer")


class SourceItem(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieved_context: List[str]
    timing_ms: float


# ========= Helpers =========

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _get_query_embedding(text: str, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def _semantic_search(
    query: str, top_k: int, threshold: float, embed_model: str
) -> List[Dict[str, Any]]:
    q_emb = _get_query_embedding(query, model=embed_model)
    scores = EMB_MATRIX @ q_emb  # cosine because both are normalized

    # pick more than top_k, then threshold + trim
    idxs = np.argsort(-scores)[: top_k * 3]
    results: List[Dict[str, Any]] = []
    for i in idxs:
        score = float(scores[i])
        if score < threshold:
            continue
        item = KB_ITEMS[i]
        results.append(
            {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "source": item["source"],
                "page": item["page"],
                "score": round(score, 4),
            }
        )
        if len(results) >= top_k:
            break
    return results


def _build_messages(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_text = "\n\n".join(
        [f"[{r['source']} p{r['page']}] {r['text']}" for r in contexts]
    )
    system = (
        "You are a compassionate mental health assistant. Use the provided context "
        "to answer the user. If no context is available, offer general, supportive "
        "guidance grounded in evidence-based self-help, and recommend professional help "
        "when appropriate. Do not diagnose. Be warm, validating, and practical."
    )
    user_msg = f"Context:\n{context_text}\n\nUser question:\n{query}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]


def _safety_footer() -> str:
    return (
        "\n\n—\nIf you’re in crisis or thinking about self-harm:\n"
        "• US: Call/text **988** (Suicide & Crisis Lifeline)\n"
        "• US Text: **HOME** to **741741** (Crisis Text Line)\n"
        "• Global centers: International Association for Suicide Prevention\n"
        "  (search: IASP Crisis Centres). If in immediate danger, call local emergency services."
    )


# ========= FastAPI Lifecycle =========

@app.on_event("startup")
def _startup():
    global KB_ITEMS, EMB_MATRIX, EMBEDDING_DIM

    if not KNOWLEDGE_BASE:
        raise RuntimeError("KNOWLEDGE_BASE is empty. Ensure knowledge_base.py is present and populated.")

    # Ensure consistent dimension; keep only rows with the modal dimension
    dims = {}
    for it in KNOWLEDGE_BASE:
        d = len(it.get("embedding", []))
        dims[d] = dims.get(d, 0) + 1
    # choose the most common dim
    EMBEDDING_DIM = max(dims, key=lambda k: dims[k])

    filtered = [it for it in KNOWLEDGE_BASE if len(it.get("embedding", [])) == EMBEDDING_DIM]

    if not filtered:
        raise RuntimeError("No embeddings with a consistent dimension found in knowledge base.")

    KB_ITEMS = filtered
    emb_matrix = np.array([it["embedding"] for it in KB_ITEMS], dtype=np.float32)
    EMB_MATRIX = _normalize_rows(emb_matrix)

    print(f"[startup] Loaded {len(KB_ITEMS)} KB items with dim={EMBEDDING_DIM}")


# ========= Endpoints =========

@app.get("/health")
def health():
    return {"status": "ok", "kb_items": len(KB_ITEMS), "embedding_dim": EMBEDDING_DIM}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    t0 = time.time()
    try:
        results = _semantic_search(
            query=payload.query,
            top_k=payload.top_k,
            threshold=payload.threshold,
            embed_model=payload.embed_model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding/search error: {e}")

    messages = _build_messages(payload.query, results)

    try:
        completion = client.chat.completions.create(
            model=payload.chat_model,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI chat error: {e}")

    if payload.safety_note:
        answer = f"{answer}{_safety_footer()}"

    timing_ms = round((time.time() - t0) * 1000.0, 2)

    return ChatResponse(
        answer=answer,
        sources=[
            SourceItem(
                chunk_id=r["chunk_id"],
                source=r["source"],
                page=r["page"],
                score=r["score"],
            )
            for r in results
        ],
        retrieved_context=[r["text"] for r in results],
        timing_ms=timing_ms,
    )
