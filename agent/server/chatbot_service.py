# chatbot_service.py
"""
chatbot_service.py
Wraps your existing chatbot implementation (chatbot.py) and exposes a simple
process_query(query, sender) -> (response_text, metadata) API.

This file intentionally keeps the heavy MeTTa/OpenAI initialization in a singleton.
"""

import os
import time
import traceback

# Import your existing chatbot code. Ensure chatbot.py is in same folder and
# provides MentalHealthMeTTaChatbot class as in your working demo.
try:
    from chatbot import MentalHealthMeTTaChatbot
except Exception as e:
    # Provide helpful diagnostic if import fails
    raise ImportError("Failed to import chatbot.MentalHealthMeTTaChatbot. Ensure chatbot.py is in the same folder.") from e

# singleton chatbot instance
_CHATBOT_INSTANCE = None

def init_chatbot_service(atomspace_file=None, embeddings_file=None):
    """Initialize singleton chatbot instance once. Safe to call multiple times."""
    global _CHATBOT_INSTANCE
    if _CHATBOT_INSTANCE is None:
        kwargs = {}
        if atomspace_file:
            kwargs["atomspace_file"] = atomspace_file
        if embeddings_file:
            kwargs["embeddings_file"] = embeddings_file
        _CHATBOT_INSTANCE = MentalHealthMeTTaChatbot(**kwargs)
    return _CHATBOT_INSTANCE

def _ensure_chatbot_initialized():
    global _CHATBOT_INSTANCE
    if _CHATBOT_INSTANCE is None:
        init_chatbot_service()
    return _CHATBOT_INSTANCE

def process_query(query: str, sender_address: str = None, timeout_seconds: int = 25):
    """
    Process query via the chatbot instance. Returns tuple (response_text, metadata).
    - metadata: dict with optional info (e.g., sources, elapsed_ms)
    This function is synchronous because uAgents server runs it in executor.
    """
    start = time.time()
    try:
        bot = _ensure_chatbot_initialized()
        # The chatbot.chat returns (response, sources)
        response_text, sources = bot.chat(query)
        elapsed_ms = int((time.time() - start) * 1000)
        metadata = {
            "elapsed_ms": elapsed_ms,
            "source_count": len(sources) if sources else 0,
            "sources": sources if len(sources) <= 5 else sources[:5]  # truncate
        }
        return response_text, metadata
    except Exception as e:
        # Log traceback to help debugging in Agentverse logs
        tb = traceback.format_exc()
        print(f"chatbot_service.process_query() error: {e}\n{tb}")
        # Return a safe fallback response (mental-health safety oriented)
        fallback = (
            "I'm sorry — I'm having trouble accessing my knowledge base right now. "
            "If you're in crisis, please call your local emergency number or a crisis line."
        )
        metadata = {"error": str(e)}
        return fallback, metadata
