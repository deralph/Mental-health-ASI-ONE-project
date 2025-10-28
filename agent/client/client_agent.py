# client_agent.py
"""
Client Agent
Sends a user query to the Server Agent and prints the response.
Designed to run on Agentverse or locally (uAgents).
"""

from uagents import Agent, Context, Model
import os

# === Configuration ===
# Default server address from your demo (change via env if needed)
SERVER_AGENT_ADDRESS = os.getenv(
    "SERVER_AGENT_ADDRESS",
    "agent1qvvcc9wxkxrhsxmk8x3lh5xzcmgm7mglz8dqs7xc0eegnhl7xfhaslc70vk"
)

agent = Agent(
    name="server",
    port=8080,  # Different port per agent (8000, 8001, etc.)
    mailbox=True,  # Enable mailbox client
    publish_agent_details=True,  # Publish to Almanac registry (CRITICAL for ASI:One)
    readme_path='./README_client_agent.md',  # Detailed README for Agentverse profile (CRITICAL)
)
# Message models
class ClientQuery(Model):
    query: str
    sender_address: str  # optional: helps server reply

class ServerReply(Model):
    response: str
    metadata: dict = None

# Send a query on startup (for demo) and also provide a helper to send programmatically
@agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info(f"Client Agent started at address: {ctx.agent.address}")
    # Demo: prompt in console when running locally (Agentverse won't prompt)
    try:
        user_query = os.getenv("DEMO_QUERY")
        if not user_query:
            # Only prompt if running interactively (local)
            user_query = input("Enter a query to send to server (or leave blank to skip): ").strip()
        if user_query:
            await ctx.send(SERVER_AGENT_ADDRESS, ClientQuery(query=user_query, sender_address=ctx.agent.address))
            ctx.logger.info(f"Sent query to server: {user_query}")
    except Exception as e:
        ctx.logger.warn(f"Startup demo failed (non-fatal): {e}")

# Receive replies from the server
@agent.on_message(model=ServerReply)
async def handle_reply(ctx: Context, sender: str, msg: ServerReply):
    ctx.logger.info(f"Reply received from {sender}")
    print("\n--- Reply from Server ---")
    print(msg.response)
    if msg.metadata:
        print("\n--- Metadata ---")
        for k, v in msg.metadata.items():
            print(f"{k}: {v}")
    print("-------------------------\n")

if __name__ == "__main__":
    agent.run()
