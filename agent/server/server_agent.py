# server_agent.py
"""
Server Agent
Receives ClientQuery messages, forwards to chatbot_service for processing,
and replies with ServerReply messages.
"""

from uagents import Agent, Context, Model
import asyncio
import os
import logging

# Use your client address(s) if you want to whitelist senders (example from demos)
KNOWN_CLIENT_ADDR = os.getenv(
    "KNOWN_CLIENT_ADDR",
    "agent1qfkvsmvnz2uvg4xa8c2j2vwzr5xyxxv0gtr4t8epvt074g3l0w8tjry48kv"
)

agent = Agent(
    name="server",
    port=8000,  # Different port per agent (8000, 8001, etc.)
    mailbox=True,  # Enable mailbox client
    publish_agent_details=True,  # Publish to Almanac registry (CRITICAL for ASI:One)
    readme_path='./README_server_agent.md',  # Detailed README for Agentverse profile (CRITICAL)
)

# Message models (mirror client models)
class ClientQuery(Model):
    query: str
    sender_address: str

class ServerReply(Model):
    response: str
    metadata: dict = None

# Import chatbot service wrapper (local module)
# Make sure chatbot_service.py is in same folder
from chatbot_service import process_query, init_chatbot_service

# Initialize chatbot service (this may load MeTTa/embeddings - done async-friendly)
init_chatbot_service()  # safe no-args init; if heavy, it preloads singleton

# Handler to accept incoming queries
@agent.on_message(model=ClientQuery)
async def handle_query(ctx: Context, sender: str, msg: ClientQuery):
    ctx.logger.info(f"Received query from {sender}: {msg.query[:80]}")

    # Optional: check sender whitelist (uncomment if desired)
    # if sender != KNOWN_CLIENT_ADDR:
    #     ctx.logger.warn(f"Unknown sender {sender} - refusing to process")
    #     await ctx.send(sender, ServerReply(response="Unauthorized sender.", metadata={"code":"unauth"}))
    #     return

    # Offload heavy processing to background task so agent mailbox isn't blocked
    async def _process_and_reply():
        try:
            # process_query is sync under the hood, so run in executor
            loop = asyncio.get_event_loop()
            result_text, metadata = await loop.run_in_executor(None, process_query, msg.query, msg.sender_address)
            # send back to sender
            await ctx.send(sender, ServerReply(response=result_text, metadata=metadata))
            ctx.logger.info("Response sent to client.")
        except Exception as e:
            ctx.logger.error(f"Processing error: {e}")
            try:
                await ctx.send(sender, ServerReply(response="Error processing your request.", metadata={"error": str(e)}))
            except Exception:
                ctx.logger.exception("Failed to send error message back to sender")

    # Start background task
    asyncio.create_task(_process_and_reply())

if __name__ == "__main__":
    # Agentverse: simply run() to attach to cloud mailbox; locally you may set env variables
    agent.run()
