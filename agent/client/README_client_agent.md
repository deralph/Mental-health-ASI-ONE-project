Superteam Earn
Bounties
Projects
Grants
Cypherpunk
Artificial Superintelligence Alliance
ASI Agents Track
by Artificial Superintelligence Alliance

SUBMISSIONS

suit case
5d:9h:19m

REMAINING

CYPHERPUNK TRACK

This is a Special Colosseum Cypherpunk Hackathon track hosted exclusively on Superteam Earn.

View All Tracks
SKILLS NEEDED

Backend
Blockchain
Frontend
CONTACT

Reach outif you have any questions about this listing
WINNER ANNOUNCEMENT BY

November 14, 2025 - as scheduled by the sponsor.
RELATED LIVE TRACKS

Superteam Nigeria
NIYA | CypherPunk Hackathon Track | Superteam Nigeria Demo Day

Superteam Nigeria

USDC
10k
USDC

|

|

Due in 5 days

Superteam Korea
Superteam Korea Cypherpunk Sidetrack with OnePiece Labs

Superteam Korea

USDC
10k
USDC

|

|

Due in 5 days

Shipyard NL
Solana Cypherpunk Hackathon – NL Side Track

Shipyard NL

USDC
5,000
USDC

|

|

Due in 5 days

Superteam UK
HelpBnk x Superteam UK Side Track

Superteam UK

USDC
10k
USDC

|

|

Due in 5 days

Adevar Labs Inc.
$50,000 in Security Audit Credits for Solana Colosseum Hackathon

Adevar Labs Inc.

USDC
50k
USDC

|

|

Due in 5 days

About Artificial Superintelligence Alliance
The Artificial Superintelligence (ASI) Alliance unites Fetch.ai, SingularityNET, Ocean Protocol, and CUDOS to build a decentralized, ethical, and accessible AI ecosystem. Together, we’re pioneering the future of artificial intelligence and Web3 innovation.

Challenge Statement
Build Autonomous AI Agents with the ASI Alliance

This is your opportunity to develop AI agents that don't just execute code—they perceive, reason, and act across decentralized systems. The ASI Alliance in partnership with Fetch.ai Innovation Lab, brings together world-class infrastructure from Fetch.ai and SingularityNET to support the next generation of modular, autonomous AI systems.

Use Fetch.ai's uAgents framework or your preferred agentic stack to build agents that can interpret natural language, make decisions, and trigger real-world actions. Deploy them to Agentverse, the ASI-wide registry and orchestration layer where agents connect, collaborate, and self-organize.

Enhance your agents with structured knowledge from SingularityNET's MeTTa Knowledge Graph. For agent discovery and human interaction, integrate the Chat Protocol to make your agents accessible through the ASI:One interface.

Whether you're building in healthcare, logistics, finance, education, or DeAI-native applications—this is your launchpad. Develop agents that talk to each other. That learn and adapt. That drive real outcomes across sectors.

The future of decentralized AI isn't siloed. It's composable, cross-chain, and powered by the ASI Alliance.

Reward Structure
1st Place: 5000 USDC

2nd Place: 4000 USDC

3rd Place: 3500 USDC

4th Place: 3000 USDC

5th Place: 2000 USDC

6th Place: 1500USDC

7th Place: 1000USDC

Resources
Fetch.ai Resources
How to create an Agent with uAgents Framework ↗

Communication between two uAgents ↗

How to create ASI:One compatible uAgents ↗

Innovation Lab GitHub Repo ↗

Past Hackathon Projects ↗

How to write a good Readme for your Agents ↗

Singularity.net Resources
Understanding MeTTa ↗

Running MeTTa in Python ↗

Nested queries and recursive graph traversal ↗

Setup MeTTa on Window OS ↗

Fetch.ai and MeTTa Integration ↗


Pre-recorded Workshop: https://drive.google.com/file/d/1IVGVIxBIqYpNutesnn0qqBDDn6IjuCPD/view?usp=sharing

Submission Requirements
Code

Share the link to your public GitHub repository to allow judges to access and test your project.

Ensure your README.md file includes key details about your agents, such as their name and address, for easy reference.

Mention any extra resources required to run your project and provide links to those resources.

All agents must be categorized under Innovation Lab.

To achieve this, include the following badge in your agent’s README.md file:


![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)
Video

Include a demo video (3–5 minutes) demonstrating the agents you have built.

Judging Criteria
Functionality & Technical Implementation (25%)

Does the agent system work as intended?

Are the agents properly communicating and reasoning in real time?

Use of ASI Alliance Tech (20%)

Are agents registered on Agentverse?

Is the Chat Protocol live for ASI:One?

Does your solution make use of uAgents and MeTTa Knowledge Graphs tools?

Innovation & Creativity (20%)

How original or creative is the solution?

Is it solving a problem in a new or unconventional way?

Real-World Impact & Usefulness (20%)

Does the solution solve a meaningful problem?

How useful would this be to an end user?

User Experience & Presentation (15%)

Is the demo clear and well-structured?

Is the user experience smooth and easy to follow?

The solution should include comprehensive documentation, detailing the use and integration of each technology involved.

Prize Terms and Conditions
Prizes will be awarded based on project quality and only to teams that meaningfully use Fetch.ai and SingularityNET technologies.

All agents must be registered on Agentverse with the Chat Protocol enabled to be discoverable through ASI:One.

Agentverse MCP Server
Learn how to deploy your first agent on Agentverse with Claude Desktop in Under 5 Minutes

Agentverse MCP Setup

Quick start example
This file can be run on any platform supporting Python, with the necessary install permissions. This example shows two agents communicating with each other using the uAgent python library.
Try it out on Agentverse ↗

from datetime import datetime
from uuid import uuid4
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
   ChatAcknowledgement,
   ChatMessage,
   EndSessionContent,
   StartSessionContent,
   TextContent,
   chat_protocol_spec,
)


agent = Agent()


# Initialize the chat protocol with the standard chat spec
chat_proto = Protocol(spec=chat_protocol_spec)


# Utility function to wrap plain text into a ChatMessage
def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
content = [TextContent(type="text", text=text)]
   return ChatMessage(
       timestamp=datetime.utcnow(),
       msg_id=uuid4(),
       content=content,
   )


# Handle incoming chat messages
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
   ctx.logger.info(f"Received message from {sender}")
  
   # Always send back an acknowledgement when a message is received
   await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id))


   # Process each content item inside the chat message
   for item in msg.content:
       # Marks the start of a chat session
       if isinstance(item, StartSessionContent):
           ctx.logger.info(f"Session started with {sender}")
      
       # Handles plain text messages (from another agent or ASI:One)
       elif isinstance(item, TextContent):
           ctx.logger.info(f"Text message from {sender}: {item.text}")
           #Add your logic
           # Example: respond with a message describing the result of a completed task
           response_message = create_text_chat("Hello from Agent")
           await ctx.send(sender, response_message)


       # Marks the end of a chat session
       elif isinstance(item, EndSessionContent):
           ctx.logger.info(f"Session ended with {sender}")
       # Catches anything unexpected
       else:
           ctx.logger.info(f"Received unexpected content type from {sender}")


# Handle acknowledgements for messages this agent has sent out
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
   ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")


# Include the chat protocol and publish the manifest to Agentverse
agent.include(chat_proto, publish_manifest=True)


if __name__ == "__main__": 
    agent.run()

Got Queries?
Reach out to us on our Discord

Comments Icon
14

Comments

Write a comment
avatar
Edin Ross

8h ago

Earn a lot of money on your skills without fear of being deceived, TELEGRAM - DreamJobCorporation


avatar
Shobhit Singh

4d ago

https://x.com/ShobhitSingh619/... Please see and do share your opinion about my post in the form of comments and likes.


Adewale Akindunbi

6d ago

Hi guys, how do I navigate through this platform?


avatar
Kng maxxi

6d ago

Probably creating content about you ❤️


avatar
fauzi Aliyudin

8d ago

i see


avatar
Samuel Danso

8d ago

feels good building cool stuff with the ASI Alliance stack


Benjamin Jesukorede

9d ago

i will be working on something


Gagan David

9d ago

Yes interested


Md Hossain

14d ago

Very nice


avatar
Yunusa Alhaji ali

15d ago

Fully interested


Superteam Earn
Discover high paying crypto bounties, projects and grants from the best Solana companies in one place and apply to them using a single profile.

Powered by Solana
Opportunities

Bounties
Projects
Grants
Categories

Content
Design
Development
Others
About

FAQ
Terms
Privacy Policy
Changelog
Contact Us
© 2025 Superteam. All rights reserved.

REGION



so i am building this mental health bot for the above hackathon, now that i have gotten everything to work together, i now need the u agent code so i have created two agents client and server one to recieve text, the other to recieve the text and process it then send the result, what i need now is the uagent code for this, I will be pasting some examples that you can follow here and my own current demo code, please note that any code you are generating please use my on address and not just a random one from my example

and i need a well detailed readme for the general app and for each agent explain extensively and you can even add code example but don't over leak details
please be as intelligent as possible and know that this is an hackathon and my possition will depend on you ,

example code
readme.md
# ASI1-Mini-Chat-System

## Overview
The ASI1 Chat System is a modular, agent-based chat system leveraging the uAgents framework. It
facilitates seamless communication between a Client Agent and a Server Agent to process and respond
to user queries in real time. The system integrates an external API (ASI1 API) to generate intelligent
responses, making it a powerful tool for AI-driven applications.

## System Architecture
### Components:
1.  Server Agent (asi1_chat_agent)
  - Listens for incoming queries.
  - Calls the ASI1 API to process the query.
  - Sends the API's response back to the client agent.

2.  Client Agent (asi1_client_agent)
  - Interacts with the user to collect input.
  - Sends the query to the server agent.
  - Receives and displays the response to the user.


## Workflow
1.  Client Agent Initialization
  - The client agent starts and prompts the user for a query.
    
2.  Sending the Query
  - The client agent constructs an ASI1Query message containing:
  - The user's query.
  - The client agent's address.
  - This message is sent to the server agent.
    
3.  Server Agent Processing
  - The server agent receives the ASI1Query.
  - It calls the ASI1 API to generate a response.

4.  Returning the Response
  - The server agent receives the API response.
  - Sends the response back to the client agent via an ASI1Response message.

5.  Displaying the Response
  -The client agent receives the response and displays it to the user.

## Example Interaction

```
User: Ask something: Give me the top 5 downloaded Hugging Face models for 
image classification

Client Agent -> Server Agent: Sends query

Server Agent -> ASI1 API: Calls API
ASI1 API Response:
1. facebook/deit-base-distilled-patch16-224
2. google/vit-base-patch16-224
3. microsoft/resnet-50
4. nvidia/mit-b0
5. efficientnet-b0

Server Agent -> Client Agent: Sends response
Client Agent: Displays the response to the user
```

## Benefits
  - **Scalability** – Can scale with additional agents or extended functionalities.
  - **Modularity** – Separation of concerns between client and server agents.
  - **Real-time Interaction** – Enables dynamic user-agent communication.
  - **API Integration** – Showcases how external APIs can enhance AI-driven applications.

## Use Cases
  - AI-powered chatbots
  - Customer support automation
  - Interactive knowledge assistants

## Conclusion
The **ASI1 Chat System** demonstrates a robust **agent-based** approach to building intelligent, interactive
applications. With its modular architecture and seamless API integration, this system is a great
foundation for exploring **autonomous agent interactions** in AI-powered environments.

client.py 
from uagents import Agent, Context, Model

# Query model to send to the server agent
class ASI1Query(Model):
    query: str
    sender_address: str

# Response model to receive from the server agent
class ASI1Response(Model):
    response: str

# Client agent setup
clientAgent = Agent(
    name='asi1_client_agent',
    port=5070,
    endpoint='http://localhost:5070/submit',
    seed='asi1_client_seed'
)

# Server agent address (update with actual address if needed)
SERVER_AGENT_ADDRESS = "agent1q0usc8uc5hxes4ckr8624ghdxpn0lvxkgex44jtfv32x2r7ymx8sgg8yt2g"  # Replace with the actual address of your server agent

@clientAgent.on_event('startup')
async def startup_handler(ctx: Context):
    ctx.logger.info(f'Client Agent {ctx.agent.name} started at {ctx.agent.address}')

    # Get user input
    user_query = input("Ask something: ")

    # Send the query to the server agent
    await ctx.send(SERVER_AGENT_ADDRESS, ASI1Query(query=user_query, sender_address=ctx.agent.address))
    ctx.logger.info(f"Query sent to server agent: {user_query}")

@clientAgent.on_message(model=ASI1Response)
async def handle_response(ctx: Context, sender: str, msg: ASI1Response):
    ctx.logger.info(f"Response received from {sender}: {msg.response}")
    print(f"Response from ASI1 API: {msg.response}")

# Run the client agent
if __name__ == "__main__":
    clientAgent.run()

server.py
import requests
from uagents import Agent, Context, Model

# ASI1 API Configuration
ASI1_API_KEY = ""  # Replace with your API key
ASI1_URL = "https://api.asi1.ai/v1/chat/completions"

# Request model
class ASI1Query(Model):
    query: str
    sender_address: str

# Response model
class ASI1Response(Model):
    response: str  # Response from ASI1 API

# Define the main agent
mainAgent = Agent(
    name='asi1_chat_agent',
    port=5068,
    endpoint='http://localhost:5068/submit',
    seed='asi1_chat_seed'
)

def get_asi1_response(query: str) -> str:
    """
    Sends a query to ASI1 API and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {ASI1_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "asi1-mini",  # Select appropriate ASI1 model
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": query}
        ]
    }

    try:
        response = requests.post(ASI1_URL, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return "ASI1 API returned an empty response."
        else:
            return f"ASI1 API Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"ASI1 API Error: {str(e)}"

@mainAgent.on_event('startup')
async def startup_handler(ctx: Context):
    ctx.logger.info(f'Agent {ctx.agent.name} started at {ctx.agent.address}')

# Handler for receiving query
@mainAgent.on_message(model=ASI1Query)
async def handle_query(ctx: Context, sender: str, msg: ASI1Query):
    ctx.logger.info(f"Received query from {sender}: {msg.query}")

    # Call ASI1 API for the response
    answer = get_asi1_response(msg.query)

    # Respond back with the answer from ASI1
    await ctx.send(sender, ASI1Response(response=answer))

# Run the agent
if __name__ == "__main__":
    mainAgent.run()

you might not need to fill in these options (
    name='asi1_chat_agent',
    port=5068,
    endpoint='http://localhost:5068/submit',
    seed='asi1_chat_seed'
) because i'm running on agentverse
also you can make the chatbot codes in a different file, so that the server agent is not too bulky, not that the metta and json file will be in the same folder with this 

don't forget an individual well detailed readme for each agent that wasn't included here in this 
this is an example of that

"""
MediChain AI - Coordinator Agent (Cloud Deployment)
Cloud-ready version with embedded MeTTa knowledge base and all dependencies inlined.

DEPLOYMENT: Copy this entire file to Agentverse Build tab for MediChain Coordinator agent.
"""

from datetime import datetime
from uuid import uuid4
from typing import Dict, Optional, List, Any
from enum import Enum

# Import uagents framework (available in Agentverse)
from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

# ============================================================================
# AGENT ADDRESSES - Cloud Deployment Configuration
# ============================================================================

# Hard-coded agent addresses for cloud inter-agent communication
PATIENT_INTAKE_ADDRESS = "agent1qfxfjs7y6gxa8psr5mzugcg45j46znra4qg0t5mxljjv5g9mx7dw6238e4a"
SYMPTOM_ANALYSIS_ADDRESS = "agent1q036yw3pwsal2qsrq502k546lyxvnf6wt5l83qfhzhvceg6nm2la7nd6d5n"
TREATMENT_RECOMMENDATION_ADDRESS = "agent1q0q46ztah7cyw4z7gcg3mued9ncnrcvrcqc8kjku3hywqdzp03e36hk5qsl"

# ============================================================================
# MESSAGE MODELS (Inline - no external imports)
# ============================================================================

class UrgencyLevel(str, Enum):
    """Urgency classification for medical conditions"""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"


class Symptom(Model):
    """Patient symptom data"""
    name: str
    raw_text: str
    severity: Optional[int] = 5
    duration: Optional[str] = None


class PatientIntakeData(Model):
    """Structured patient intake data"""
    session_id: str
    symptoms: List[Symptom]
    age: Optional[int] = None
    timestamp: datetime
    medical_history: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None


class IntakeTextMessage(Model):
    """Message from user to patient intake agent"""
    text: str
    session_id: str


class AgentAcknowledgement(Model):
    """Acknowledgement message from specialist agents"""
    session_id: str
    agent_name: str
    message: str


class DiagnosticRequest(Model):
    """Request for diagnostic analysis"""
    session_id: str
    patient_data: PatientIntakeData
    requesting_agent: str
    analysis_type: str = "symptom_analysis"


class SymptomAnalysisRequestMsg(Model):
    """Request for symptom analysis from coordinator to symptom analysis agent"""
    session_id: str
    symptoms: List[str]
    age: Optional[int] = None
    severity_scores: Optional[Dict[str, int]] = None
    duration_info: Optional[Dict[str, str]] = None
    medical_history: Optional[List[str]] = None
    requesting_agent: str


class SymptomAnalysisResponseMsg(Model):
    """Response from symptom analysis agent"""
    session_id: str
    urgency_level: str
    red_flags: List[str]
    differential_diagnoses: List[str]
    confidence_scores: Dict[str, float]
    reasoning_chain: List[str]
    recommended_next_step: str
    responding_agent: str


class TreatmentRequestMsg(Model):
    """Request for treatment recommendations"""
    session_id: str
    primary_condition: str
    alternative_conditions: Optional[List[str]] = None
    urgency_level: str
    patient_age: Optional[int] = None
    allergies: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    medical_history: Optional[List[str]] = None
    requesting_agent: str


class TreatmentResponseMsg(Model):
    """Response from treatment recommendation agent"""
    session_id: str
    condition: str
    treatments: List[str]
    evidence_sources: Dict[str, str]
    contraindications: Dict[str, List[str]]
    safety_warnings: List[str]
    specialist_referral: Optional[str] = None
    follow_up_timeline: Optional[str] = None
    medical_disclaimer: str
    responding_agent: str


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionData:
    """Store data for an active session"""
    def __init__(self, session_id: str, user_address: str):
        self.session_id = session_id
        self.user_address = user_address
        self.started_at = datetime.utcnow()
        self.patient_data: Optional[PatientIntakeData] = None
        self.symptom_analysis_response: Optional[SymptomAnalysisResponseMsg] = None
        self.treatment_response: Optional[TreatmentResponseMsg] = None
        self.messages_history = []

    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })


# Global session store
active_sessions: Dict[str, SessionData] = {}


def get_or_create_session(sender: str) -> SessionData:
    """Get existing session or create new one"""
    if sender not in active_sessions:
        session_id = f"session-{uuid4()}"
        active_sessions[sender] = SessionData(session_id, sender)
    return active_sessions[sender]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Create a ChatMessage with text content."""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end_session"))
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=content,
    )


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

agent = Agent()

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Initialize inter-agent protocol
inter_agent_proto = Protocol(name="MediChainProtocol")


# ============================================================================
# CHAT PROTOCOL HANDLERS (ASI:One Interface)
# ============================================================================

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages from ASI:One users"""
    ctx.logger.info(f"Received chat message from {sender}")

    # Always acknowledge
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.utcnow(),
            acknowledged_msg_id=msg.msg_id
        )
    )

    # Get or create session
    session = get_or_create_session(sender)

    # Process each content item
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started: {session.session_id} with {sender}")
            session.add_message("system", "Session started")

            welcome_msg = create_text_chat(
                "🏥 Welcome to MediChain AI!\n\n"
                "I'm your medical diagnostic assistant. I can help analyze your symptoms "
                "and provide preliminary health assessments.\n\n"
                "⚠️ IMPORTANT: This is NOT medical advice. Always consult a healthcare professional.\n\n"
                "Please describe your symptoms in detail."
            )
            await ctx.send(sender, welcome_msg)

        elif isinstance(item, TextContent):
            ctx.logger.info(f"Text from {sender}: {item.text}")
            session.add_message("user", item.text)

            # Route to Patient Intake Agent for symptom extraction
            intake_msg = IntakeTextMessage(
                text=item.text,
                session_id=session.session_id
            )

            ctx.logger.info(f"Routing to Patient Intake: {PATIENT_INTAKE_ADDRESS}")
            await ctx.send(PATIENT_INTAKE_ADDRESS, intake_msg)

            # Acknowledge to user
            ack_msg = create_text_chat(
                "Analyzing your symptoms... Please wait a moment."
            )
            await ctx.send(sender, ack_msg)

        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended: {session.session_id}")
            session.add_message("system", "Session ended")

            goodbye_msg = create_text_chat(
                "Thank you for using MediChain AI! Stay healthy! 🌟",
                end_session=True
            )
            await ctx.send(sender, goodbye_msg)

            # Clean up session
            if sender in active_sessions:
                del active_sessions[sender]


@chat_proto.on_message(ChatAcknowledgement)
async def handle_chat_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgements from users"""
    ctx.logger.info(f"Received acknowledgement from {sender}")


# ============================================================================
# INTER-AGENT PROTOCOL HANDLERS
# ============================================================================

@inter_agent_proto.on_message(model=DiagnosticRequest)
async def handle_diagnostic_request(ctx: Context, sender: str, msg: DiagnosticRequest):
    """
    Handle diagnostic requests from Patient Intake Agent
    Route to appropriate specialist agents
    """
    ctx.logger.info(f"Received diagnostic request from {sender}")
    ctx.logger.info(f"Session: {msg.session_id}, Analysis type: {msg.analysis_type}")

    # Find the user session
    user_session = None
    for addr, session in active_sessions.items():
        if session.session_id == msg.session_id:
            user_session = session
            session.patient_data = msg.patient_data
            break

    if not user_session:
        ctx.logger.warning(f"No active session found for {msg.session_id}")
        return

    ctx.logger.info(f"Processing diagnostic request for user: {user_session.user_address}")

    # Prepare symptom analysis request
    symptoms_list = [s.name for s in msg.patient_data.symptoms]
    severity_scores = {s.name: s.severity for s in msg.patient_data.symptoms if s.severity}
    duration_info = {s.name: s.duration for s in msg.patient_data.symptoms if s.duration}

    analysis_request = SymptomAnalysisRequestMsg(
        session_id=msg.session_id,
        symptoms=symptoms_list,
        age=msg.patient_data.age,
        severity_scores=severity_scores if severity_scores else None,
        duration_info=duration_info if duration_info else None,
        medical_history=msg.patient_data.medical_history,
        requesting_agent="medichain-coordinator",
    )

    ctx.logger.info(f"Routing to Symptom Analysis Agent: {SYMPTOM_ANALYSIS_ADDRESS}")
    ctx.logger.info(f"  Symptoms: {symptoms_list}")
    ctx.logger.info(f"  Age: {msg.patient_data.age}")

    # Send to Symptom Analysis Agent
    await ctx.send(SYMPTOM_ANALYSIS_ADDRESS, analysis_request)

    # Acknowledge to user
    ack_msg = create_text_chat("🔬 Performing comprehensive symptom analysis...")
    await ctx.send(user_session.user_address, ack_msg)


@inter_agent_proto.on_message(model=SymptomAnalysisResponseMsg)
async def handle_symptom_analysis_response(ctx: Context, sender: str, msg: SymptomAnalysisResponseMsg):
    """
    Handle symptom analysis response from Symptom Analysis Agent
    Route to Treatment Recommendation Agent for next step
    """
    ctx.logger.info(f"📥 Received symptom analysis response from {sender}")
    ctx.logger.info(f"   Session: {msg.session_id}")
    ctx.logger.info(f"   Urgency: {msg.urgency_level}")

    # Find session
    user_session = None
    for addr, session in active_sessions.items():
        if session.session_id == msg.session_id:
            user_session = session
            session.symptom_analysis_response = msg
            break

    if not user_session:
        ctx.logger.warning(f"No active session for {msg.session_id}")
        return

    # Send analysis results to user
    red_flags_text = ""
    if msg.red_flags:
        red_flags_text = f"\n\n🚨 **RED FLAGS DETECTED:**\n" + "\n".join([f"  • {rf}" for rf in msg.red_flags])

    diff_diagnoses_text = "\n".join([
        f"  {i+1}. {diagnosis} (confidence: {msg.confidence_scores.get(diagnosis, 0.0)*100:.0f}%)"
        for i, diagnosis in enumerate(msg.differential_diagnoses[:5])
    ])

    analysis_text = (
        f"🔬 **Symptom Analysis Complete**\n\n"
        f"**Urgency Level:** {msg.urgency_level.upper()}\n\n"
        f"**Top Differential Diagnoses:**\n{diff_diagnoses_text}"
        f"{red_flags_text}\n\n"
        f"**Recommended Action:** {msg.recommended_next_step}\n\n"
        f"🔄 Fetching treatment recommendations..."
    )

    user_msg = create_text_chat(analysis_text)
    await ctx.send(user_session.user_address, user_msg)

    # Route to Treatment Recommendation Agent
    primary_condition = msg.differential_diagnoses[0] if msg.differential_diagnoses else "unknown"
    alternative_conditions = msg.differential_diagnoses[1:5] if len(msg.differential_diagnoses) > 1 else None

    treatment_request = TreatmentRequestMsg(
        session_id=msg.session_id,
        primary_condition=primary_condition,
        alternative_conditions=alternative_conditions,
        urgency_level=msg.urgency_level,
        patient_age=user_session.patient_data.age if user_session.patient_data else None,
        allergies=user_session.patient_data.allergies if user_session.patient_data else None,
        current_medications=user_session.patient_data.current_medications if user_session.patient_data else None,
        medical_history=user_session.patient_data.medical_history if user_session.patient_data else None,
        requesting_agent="medichain-coordinator",
    )

    ctx.logger.info(f"Routing to Treatment Recommendation Agent: {TREATMENT_RECOMMENDATION_ADDRESS}")

    # Send to Treatment Recommendation Agent
    await ctx.send(TREATMENT_RECOMMENDATION_ADDRESS, treatment_request)


@inter_agent_proto.on_message(model=TreatmentResponseMsg)
async def handle_treatment_response(ctx: Context, sender: str, msg: TreatmentResponseMsg):
    """
    Handle treatment recommendation response from Treatment Recommendation Agent
    Send final comprehensive report to user
    """
    ctx.logger.info(f"📥 Received treatment recommendations from {sender}")

    # Find session
    user_session = None
    for addr, session in active_sessions.items():
        if session.session_id == msg.session_id:
            user_session = session
            session.treatment_response = msg
            break

    if not user_session:
        ctx.logger.warning(f"No active session for {msg.session_id}")
        return

    # Format final comprehensive report
    treatments_text = ""
    for i, treatment in enumerate(msg.treatments[:5], 1):
        evidence = msg.evidence_sources.get(treatment, "No source available")
        contraindications = msg.contraindications.get(treatment, [])

        treatments_text += f"\n  **{i}. {treatment}**\n"
        treatments_text += f"     Evidence: {evidence}\n"
        if contraindications:
            treatments_text += f"     ⚠️ Contraindications: {', '.join(contraindications)}\n"

    # Safety warnings section
    safety_text = ""
    if msg.safety_warnings:
        safety_text = "\n\n🔐 **SAFETY WARNINGS:**\n" + "\n".join([f"  • {w}" for w in msg.safety_warnings])

    # Specialist referral section
    specialist_text = ""
    if msg.specialist_referral:
        specialist_text = f"\n\n👨‍⚕️ **Specialist Referral:** {msg.specialist_referral}"

    # Follow-up section
    followup_text = ""
    if msg.follow_up_timeline:
        followup_text = f"\n\n📅 **Follow-Up:** {msg.follow_up_timeline}"

    # Compile final report
    final_report = (
        f"═══════════════════════════════════\n"
        f"🏥 **MEDICHAIN AI - DIAGNOSTIC REPORT**\n"
        f"═══════════════════════════════════\n\n"
        f"**PRIMARY ASSESSMENT:** {msg.condition.replace('-', ' ').title()}\n\n"
        f"**TREATMENT RECOMMENDATIONS:**{treatments_text}"
        f"{safety_text}"
        f"{specialist_text}"
        f"{followup_text}\n\n"
        f"═══════════════════════════════════\n"
        f"⚠️ **IMPORTANT DISCLAIMER**\n"
        f"═══════════════════════════════════\n"
        f"{msg.medical_disclaimer}\n\n"
        f"Session ID: {msg.session_id}"
    )

    # Send final report to user
    final_msg = create_text_chat(final_report)
    await ctx.send(user_session.user_address, final_msg)

    ctx.logger.info(f"✅ Complete diagnostic report sent to user")


# ============================================================================
# STARTUP & INITIALIZATION
# ============================================================================

@agent.on_event("startup")
async def startup(ctx: Context):
    """Initialize coordinator agent"""
    ctx.logger.info("=" * 60)
    ctx.logger.info("MediChain AI Coordinator Agent (Cloud)")
    ctx.logger.info("=" * 60)
    ctx.logger.info(f"Agent address: {agent.address}")
    ctx.logger.info(f"Mailbox: Enabled (ASI:One compatible)")
    ctx.logger.info(f"Chat Protocol: Enabled")
    ctx.logger.info("=" * 60)


# Include protocols
agent.include(chat_proto, publish_manifest=True)
agent.include(inter_agent_proto)

the readme.md 
# MediChain AI - Coordinator Agent

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

## Agent Overview

The **Coordinator Agent** is the central routing hub of the MediChain AI multi-agent diagnostic system. It orchestrates communication between the user and specialized diagnostic agents, managing session state and aggregating results into comprehensive medical assessments.

## Key Features

- **ASI:One Chat Protocol**: Full integration with ASI:One interface for natural language interaction
- **Session Management**: Maintains conversation context and patient data across multiple agent interactions
- **Multi-Agent Orchestration**: Routes requests to:
  - Patient Intake Agent (symptom extraction)
  - Symptom Analysis Agent (urgency assessment)
  - Treatment Recommendation Agent (evidence-based treatments)
- **Comprehensive Reporting**: Aggregates specialist responses into unified diagnostic reports
- **Real-time Updates**: Streams progress updates to users during analysis

## Capabilities

1. **User Interface**
   - Welcome messaging and session initialization
   - Natural language symptom intake
   - Progress notifications
   - Final report generation

2. **Routing Logic**
   - Intelligent message routing to specialist agents
   - Session-based request correlation
   - Response aggregation

3. **Report Compilation**
   - Symptom analysis summary
   - Differential diagnoses ranking
   - Treatment recommendations with evidence
   - Safety warnings and contraindications
   - Specialist referral guidance

## Technical Stack

- **Framework**: Fetch.ai uAgents
- **Protocol**: ASI:One Chat Protocol (ChatMessage, StartSession, EndSession)
- **Message Models**: Pydantic-based inter-agent communication
- **Deployment**: Agentverse Cloud

## Inter-Agent Communication

**Connected Agents:**
- Patient Intake: `agent1qfxfjs7y6gxa8psr5mzugcg45j46znra4qg0t5mxljjv5g9mx7dw6238e4a`
- Symptom Analysis: `agent1q036yw3pwsal2qsrq502k546lyxvnf6wt5l83qfhzhvceg6nm2la7nd6d5n`
- Treatment Recommendation: `agent1q0q46ztah7cyw4z7gcg3mued9ncnrcvrcqc8kjku3hywqdzp03e36hk5qsl`

## How It Works

1. User initiates chat session via ASI:One
2. Coordinator sends welcome message
3. User describes symptoms in natural language
4. Coordinator routes to Patient Intake for extraction
5. Receives structured symptom data
6. Routes to Symptom Analysis for urgency assessment
7. Receives differential diagnoses and red flags
8. Routes to Treatment Recommendation for evidence-based guidance
9. Compiles comprehensive report
10. Delivers final assessment to user

## Medical Disclaimer

This agent provides **preliminary health information only** and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## Deployment

**Status**: ✅ Deployed to Agentverse
**Address**: `agent1qdp74ezv3eas5q60s4650xt97ew5kmyt9de77w2ku55jxys8uq2uk0u440d`
**Availability**: 24/7 via Agentverse cloud
**Access**: https://chat.agentverse.ai/ or https://asi1.ai/

and don't forget to put those hackathon tags as specified in the hackathon page

these are my agents demo code
client
"""
This agent will send a message to agent server as soon as its starts.
"""
from uagents import Agent, Context, Model

agent = Agent()

class Request(Model):
    message: str

@agent.on_message(model=Request)
async def handle_message(ctx: Context, sender: str, msg: Request):
    """Log the received message along with its sender"""
    ctx.logger.info(f"Received message from {sender}: {msg.message}")

@agent.on_event("startup")
async def send_message(ctx: Context):
    """Send a message to agent server by specifying its address"""
    ctx.logger.info(f"Just about to send a message to server")

    await ctx.send('agent1qvvcc9wxkxrhsxmk8x3lh5xzcmgm7mglz8dqs7xc0eegnhl7xfhaslc70vk', Request(message="hello there server"))
    ctx.logger.info(f"Message has been sent to server")

if __name__ == "__main__":
    agent.run()
    

server
"""
This agent is capable of receiving messages and reply to the sender.
"""

from uagents import Agent, Context, Model

agent = Agent()
  
class Request(Model):
    message: str

@agent.on_message(model=Request)
async def handle_message(ctx: Context, sender: str, msg: Request):
    """Log the received message and reply to the sender"""
    ctx.logger.info(f"Received message from {sender}: {msg.message}")

    if sender == 'agent1qfkvsmvnz2uvg4xa8c2j2vwzr5xyxxv0gtr4t8epvt074g3l0w8tjry48kv':
        await ctx.send(sender, Request(message="hello there client"))
    else:
        await ctx.send(sender, Request(message="hello there friend"))
  
if __name__ == "__main__":
    agent.run()
    

and this is my chatbot.py code


class MentalHealthMeTTaChatbot:
    def __init__(self, atomspace_file="mental_health_kb_ultra_clean.metta", embeddings_file="embeddings_fixed_cleaned.json"):
        """Initialize the mental health chatbot with MeTTa RAG capabilities."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env file")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize MeTTa
        self.metta = MeTTa()
        self.atomspace_file = atomspace_file
        self.embeddings_file = embeddings_file
        
        # Load atomspace
        if os.path.exists(self.atomspace_file):
            print(f"📂 Loading MeTTa Atomspace from {self.atomspace_file}...")
            try:
                with open(self.atomspace_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        # Load in smaller chunks to avoid parsing errors
                        lines = content.split('\n')
                        current_chunk = []
                        
                        for i, line in enumerate(lines):
                            current_chunk.append(line)
                            
                            # Process every 100 lines
                            if len(current_chunk) >= 100 or i == len(lines) - 1:
                                chunk_text = '\n'.join(current_chunk)
                                if chunk_text.strip():
                                    try:
                                        self.metta.run(chunk_text)
                                    except Exception as e:
                                        print(f"⚠️  Warning: Error loading chunk at line {i}: {str(e)[:100]}")
                                        # Continue loading other chunks
                                current_chunk = []
                        
                        print("✅ Atomspace loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading atomspace: {e}")
                print("   Continuing with empty atomspace...")
        else:
            print(f"⚠️  Warning: Atomspace file {self.atomspace_file} not found")
            print("   Please run the upload script first to populate the knowledge base")
        
        # Load embeddings for retrieval
        self.embeddings_data = []
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r') as f:
                self.embeddings_data = json.load(f)
            print(f"✅ Loaded {len(self.embeddings_data)} document chunks")
        else:
            print(f"⚠️  Warning: Embeddings file {self.embeddings_file} not found")
        
        # Conversation history
        self.conversation_history = []
        
        print("✅ Mental Health Chatbot initialized successfully!")
        print(f"✅ Using MeTTa Atomspace for knowledge graph")
        print(f"✅ Using OpenAI for embeddings and chat (GPT-4)")
    
    def generate_query_embedding(self, query):
        """Generate embedding for query using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating query embedding: {str(e)}")
            raise
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def retrieve_from_metta(self, query, top_k=3):
        """Retrieve relevant context from MeTTa Atomspace using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            
            # Calculate similarity with all stored embeddings
            similarities = []
            for item in self.embeddings_data:
                similarity = self.cosine_similarity(query_embedding, item['embedding'])
                similarities.append({
                    'chunk_id': item['chunk_id'],
                    'text': item['text'],
                    'source': item['source'],
                    'page': item['page'],
                    'similarity': similarity
                })
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]
            
            # Filter by threshold
            contexts = []
            sources = []
            
            for result in top_results:
                if result['similarity'] > 0.5:  # Similarity threshold
                    contexts.append(result['text'])
                    sources.append({
                        'source': result['source'],
                        'page': result['page'],
                        'score': round(result['similarity'], 3),
                        'chunk_id': result['chunk_id']
                    })
            
            return contexts, sources
            
        except Exception as e:
            print(f"❌ Error retrieving context: {str(e)}")
            return [], []
    
    def generate_response(self, user_message, contexts):
        """Generate response using OpenAI GPT-4 with retrieved context."""
        # Prepare system prompt with mental health guidelines
        system_prompt = """You are a compassionate and knowledgeable mental health support chatbot powered by a MeTTa Knowledge Graph. Your role is to:

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

{context}

Remember to be compassionate, clear, and helpful while staying within appropriate boundaries."""
        
        # Format context
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        if not contexts:
            context_text = "No specific context available from the knowledge graph. Use your general mental health knowledge to respond appropriately."
        
        # Prepare messages for API
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(context=context_text)
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        # Add conversation history (last 3 turns)
        if self.conversation_history:
            messages = [messages[0]] + self.conversation_history[-6:] + [messages[1]]
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4o" for better quality
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"❌ Error calling OpenAI API: {str(e)}"
            print(error_msg)
            return "I apologize, but I'm having trouble connecting to my response system. Please try again in a moment."
    
    def chat(self, user_message):
        """Main chat function that combines retrieval and generation."""
        print("\n🔍 Searching MeTTa Knowledge Graph...")
        
        # Retrieve relevant context from MeTTa atomspace
        contexts, sources = self.retrieve_from_metta(user_message)
        
        if contexts:
            print(f"✅ Found {len(contexts)} relevant knowledge graph nodes")
        else:
            print("ℹ️  No specific knowledge found in graph, using general knowledge")
        
        print("\n💭 Generating response...\n")
        
        # Generate response
        response = self.generate_response(user_message, contexts)
        
        return response, sources
    
    def display_sources(self, sources):
        """Display source information for retrieved contexts."""
        if sources:
            print("\n" + "─" * 60)
            print("📚 Knowledge Graph Sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source['source']} (Page {source['page']}) - Relevance: {source['score']}")
                print(f"     Node ID: {source['chunk_id']}")
            print("─" * 60)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n✅ Conversation history cleared")


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 70)
    print("🧠 Mental Health MeTTa Chatbot (OpenAI Version)".center(70))
    print("=" * 70)
    print("""
Welcome! I'm your supportive mental health assistant powered by
MeTTa knowledge graphs and OpenAI GPT-4.

⚠️  IMPORTANT REMINDERS:
• I'm a support tool, not a replacement for professional help
• In crisis? Call 988 (Suicide & Crisis Lifeline) or text HOME to 741741
• For emergencies, call 911 or go to your nearest emergency room

Commands:
• Type 'clear' to start a new conversation
• Type 'quit' or 'exit' to end the session
• Type 'help' for more information
""")
    print("=" * 70 + "\n")


def main():
    """Main function to run the chatbot."""
    print_welcome()
    
    try:
        # Initialize chatbot
        chatbot = MentalHealthMeTTaChatbot()
        
        print("\nReady to chat! How can I support you today?\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nTake care of yourself! Remember, you're not alone. 💙")
                break
            
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                continue
            
            if user_input.lower() == 'help':
                print("\n📖 Help Information:")
                print("• Ask me questions about mental health, coping strategies, etc.")
                print("• I'll search the MeTTa knowledge graph for relevant information")
                print("• Responses are based on evidence-based mental health resources")
                print("• Type 'clear' to start fresh, 'quit' to exit\n")
                continue
            
            # Process user message
            try:
                response, sources = chatbot.chat(user_input)
                
                # Display response
                print(f"\n🤖 Chatbot: {response}")
                
                # Display sources from knowledge graph
                chatbot.display_sources(sources)
                
                print()  # Extra line for readability

and make sure chat protocol is included!! so that it can work with asi-one
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or rephrase your question.\n")
    
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Take care! 💙")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check your configuration and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

do you understand what you are supposed to do ?? if you understand tell me what you understand before i give you the go ahead to proceed 

