# chatbot_no_hyperon.py
import os, sys, json
import numpy as np
from openai import OpenAI
from typing import List
from knowledge_base import KNOWLEDGE_BASE, CHUNK_INDEX

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment before running chatbot")

client = OpenAI(api_key=OPENAI_API_KEY)

# Precompute embeddings matrix for fast similarity (loads into memory)
EMBEDDING_DIM = len(KNOWLEDGE_BASE[0]['embedding']) if KNOWLEDGE_BASE else 1536
emb_matrix = np.array([item['embedding'] for item in KNOWLEDGE_BASE], dtype=np.float32)
# Normalize for cosine similarity
norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
# avoid divide-by-zero
norms[norms == 0] = 1.0
emb_matrix = emb_matrix / norms

def get_query_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    q = np.array(resp.data[0].embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return q
    return q / q_norm

def semantic_search(query: str, top_k: int = 3, threshold: float = 0.0):
    q_emb = get_query_embedding(query)
    # dot product since both normalized -> cosine
    scores = emb_matrix @ q_emb
    # get top_k indices
    idxs = np.argsort(-scores)[:top_k*3]  # pick a few more to filter threshold
    results = []
    for i in idxs:
        score = float(scores[i])
        if score < threshold:
            continue
        item = KNOWLEDGE_BASE[i]
        results.append({
            "chunk_id": item['chunk_id'],
            "text": item['text'],
            "source": item['source'],
            "page": item['page'],
            "score": round(score, 4)
        })
        if len(results) >= top_k:
            break
    return results

# Example usage — interactive CLI
def chat_loop():
    print("Mental Health Chatbot (no hyperon) — type 'quit' to exit")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in ("quit","exit","bye"):
            print("Goodbye — take care.")
            break
        results = semantic_search(q, top_k=3, threshold=0.0)
        if results:
            print("\nRelevant sources:")
            for r in results:
                print(f" - ({r['source']} p{r['page']}) score={r['score']}")
                snippet = (r['text'] or "")[:800]
                print(snippet)
                print("----")
        else:
            print("No relevant chunks found. Answering from general knowledge...\n")
        # optional: call OpenAI chat for final response combining context
        # build prompt:
        ctx = "\n\n".join([f"[{r['source']} p{r['page']}] {r['text']}" for r in results])
        system = ("You are a compassionate mental health assistant. Use the provided context to answer the user. "
                  "If no context is available, answer with general supportive guidance and recommend professional help when appropriate.")
        messages = [
            {"role":"system","content":system},
            {"role":"user","content": f"Context:\n{ctx}\n\nQuestion:\n{q}"}
        ]
        try:
            chat_resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=800)
            answer = chat_resp.choices[0].message.content
            print("\nChatbot:", answer)
        except Exception as e:
            print("Error calling OpenAI chat:", e)
            # fallback simple reply
            print("Chatbot: I can help with supportive resources. Consider seeking in-person professional help for severe symptoms.")

if __name__ == "__main__":
    chat_loop()