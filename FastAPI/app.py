import os
import json
import math
from typing import List, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# MeTTa/Hyperon only for loading .metta at startup or /reload
from hyperon import MeTTa

# ----------------------------
# Environment & Globals
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

DEFAULT_METTA_PATH = os.getenv("METTA_PATH", "mental_health_kb_ultra_clean.metta")
DEFAULT_EMBED_PATH = os.getenv("EMBED_PATH", "embeddings_fixed_cleaned.json")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")     # 1536 dims
DEFAULT_CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

# App + OpenAI client
app = FastAPI(title="Mental Health MeTTa Chatbot API", version="1.0.0")
client = OpenAI(api_key=OPENAI_API_KEY)

# In-memory KB / state
metta = None
embeddings_data: List[Dict[str, Any]] = []
dim_buckets: Dict[int, List[int]] = defaultdict(list)  # dim -> indices in embeddings_data
conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)  # session_id -> last N messages
MAX_TURNS = 6  # keep last 3 user/assistant pairs

# ----------------------------
# Models
# ----------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    session_id: Optional[str] = Field(default="default", description="Session id for short-term memory")
    top_k: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    embed_model: Optional[str] = Field(default=None, description="Override embedding model")
    chat_model: Optional[str] = Field(default=None, description="Override chat model")

class SourceItem(BaseModel):
    source: str
    page: int
    score: float
    chunk_id: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceItem] = []
    used_models: Dict[str, str] = {}
    tokens_hint: Optional[Dict[str, int]] = None

class ReloadRequest(BaseModel):
    metta_path: Optional[str] = None
    embeddings_path: Optional[str] = None

# ----------------------------
# Utilities
# ----------------------------
def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def load_metta_atomspace(metta_path: str) -> MeTTa:
    m = MeTTa()
    if not os.path.exists(metta_path):
        print(f"⚠️  MeTTa file not found: {metta_path}")
        return m
    print(f"📂 Loading MeTTa Atomspace from {metta_path} ...")
    with open(metta_path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        print("ℹ️  MeTTa file empty")
        return m

    # Chunked loading to reduce parse failures
    lines = content.splitlines()
    chunk = []
    for i, line in enumerate(lines, 1):
        chunk.append(line)
        if len(chunk) >= 200 or i == len(lines):
            block = "\n".join(chunk)
            if block.strip():
                try:
                    m.run(block)
                except Exception as e:
                    # Best-effort load: continue after a bad block
                    print(f"⚠️  MeTTa parse warning near line {i}: {str(e)[:150]}")
            chunk = []
    print("✅ Atomspace loaded")
    return m

def load_embeddings(emb_path: str):
    if not os.path.exists(emb_path):
        print(f"⚠️  Embeddings file not found: {emb_path}")
        return []

    with open(emb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize shapes and build dimension buckets
    dim_buckets.clear()
    for idx, item in enumerate(data):
        vec = item.get("embedding")
        if not isinstance(vec, list) or len(vec) == 0:
            continue
        dim_buckets[len(vec)].append(idx)

    # Small summary
    counts = {dim: len(idxs) for dim, idxs in dim_buckets.items()}
    print(f"Embedding dimension distribution: {counts}")
    return data

def make_query_embedding(text: str, model: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def select_bucket_indices(query_dim: int) -> List[int]:
    """Return indices for the embeddings that match query_dim.
    If none match, fall back to the largest bucket (majority dimension)."""
    if query_dim in dim_buckets and dim_buckets[query_dim]:
        return dim_buckets[query_dim]
    # Fallback to the largest bucket to avoid total failure
    best_dim = max(dim_buckets.keys(), key=lambda d: len(dim_buckets[d])) if dim_buckets else None
    return dim_buckets.get(best_dim, [])

def retrieve(query: str, top_k: int, threshold: float, embed_model: str):
    qvec = np.array(make_query_embedding(query, embed_model), dtype=np.float32)
    qdim = qvec.shape[0]
    candidate_indices = select_bucket_indices(qdim)

    if not candidate_indices:
        return [], []

    sims = []
    for idx in candidate_indices:
        vec = embeddings_data[idx]["embedding"]
        # shape already assured by bucket
        sim = safe_cosine(qvec, np.array(vec, dtype=np.float32))
        sims.append((idx, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:top_k]

    contexts, sources = [], []
    for idx, score in top:
        if score < threshold:
            continue
        item = embeddings_data[idx]
        contexts.append(item.get("text", ""))  # may be empty if embeddings.json lacked text
        sources.append(SourceItem(
            source=item.get("source", "unknown"),
            page=int(item.get("page", 0)),
            score=round(float(score), 3),
            chunk_id=item.get("chunk_id", "unknown")
        ))
    return contexts, sources

def build_system_prompt(contexts: List[str]) -> str:
    if contexts:
        context_text = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
    else:
        context_text = ("No specific context available from the knowledge graph. "
                        "Use general mental health knowledge to respond appropriately.")

    return f"""You are a compassionate and knowledgeable mental health support chatbot powered by a MeTTa Knowledge Graph. Your role is to:

1. Provide supportive, empathetic responses based on evidence-based mental health information
2. Use the provided context from mental health resources stored in the MeTTa Atomspace to inform your answers
3. Always prioritize user safety and well-being
4. Encourage professional help when appropriate (crisis situations, severe symptoms)
5. Never diagnose or replace professional mental health services
6. Be warm, non-judgmental, and validating
7. Provide practical coping strategies when relevant

IMPORTANT SAFETY GUIDELINES:
- If the user mentions suicidal thoughts, self-harm, or crisis: Immediately provide crisis resources:
  * National Suicide Prevention Lifeline: 988 (US)
  * Crisis Text Line: Text HOME to 741741 (US)
  * International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- For severe mental health symptoms: Encourage seeking professional help
- You are a support tool, not a replacement for therapy or psychiatric care

Use the following context retrieved from the MeTTa Knowledge Graph:

{context_text}

Remember to be compassionate, clear, and helpful while staying within appropriate boundaries.
"""

def chat_completion(messages: List[Dict[str, str]], model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=900
    )
    return resp.choices[0].message.content

def trim_history(session_id: str):
    if len(conversation_history[session_id]) > MAX_TURNS:
        conversation_history[session_id] = conversation_history[session_id][-MAX_TURNS:]

# ----------------------------
# Startup loader
# ----------------------------
def boot_load(metta_path: str, emb_path: str):
    global metta, embeddings_data
    metta = load_metta_atomspace(metta_path)
    embeddings_data = load_embeddings(emb_path)
    print(f"✅ Ready: {len(embeddings_data)} chunks")

boot_load(DEFAULT_METTA_PATH, DEFAULT_EMBED_PATH)

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/reload")
def reload_data(req: ReloadRequest):
    metta_path = req.metta_path or DEFAULT_METTA_PATH
    emb_path = req.embeddings_path or DEFAULT_EMBED_PATH
    boot_load(metta_path, emb_path)
    return {"status": "reloaded",
            "metta_path": metta_path,
            "embeddings_path": emb_path,
            "chunks": len(embeddings_data),
            "dims": {dim: len(idxs) for dim, idxs in dim_buckets.items()}}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    embed_model = req.embed_model or DEFAULT_EMBED_MODEL
    chat_model  = req.chat_model  or DEFAULT_CHAT_MODEL

    # Retrieve context
    contexts, sources = retrieve(
        query=req.message,
        top_k=req.top_k,
        threshold=req.similarity_threshold,
        embed_model=embed_model
    )

    # Build messages with short session memory
    system_prompt = build_system_prompt(contexts)
    history = conversation_history[req.session_id]
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": req.message}
    ]

    # Generate
    try:
        reply = chat_completion(messages, chat_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    # Update memory
    conversation_history[req.session_id].append({"role": "user", "content": req.message})
    conversation_history[req.session_id].append({"role": "assistant", "content": reply})
    trim_history(req.session_id)

    return ChatResponse(
        reply=reply,
        sources=sources,
        used_models={"chat": chat_model, "embedding": embed_model}
    )
