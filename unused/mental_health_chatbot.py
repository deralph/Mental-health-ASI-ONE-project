"""
Mental Health Chatbot RAG - Main Chatbot Application
This script creates an interactive chatbot that retrieves relevant information from Pinecone
and generates responses using OpenRouter.

Required packages:
pip install pinecone-client openai python-dotenv langchain langchain-openai requests

Required .env file:
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key (for embeddings)
OPENROUTER_API_KEY=your_openrouter_api_key
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import requests
import json

# Load environment variables
load_dotenv()

class MentalHealthChatbot:
    def __init__(self):
        """Initialize the mental health chatbot with RAG capabilities."""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not all([self.pinecone_api_key, self.openai_api_key, self.openrouter_api_key]):
            raise ValueError("Missing required API keys in .env file")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "mental-health-chatbot"
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # OpenRouter configuration
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        # You can change this model to any available on OpenRouter
        self.model = "anthropic/claude-3.5-sonnet"  # or "openai/gpt-4", etc.
        
        # Conversation history
        self.conversation_history = []
        
        print("✓ Mental Health Chatbot initialized successfully!")
        print(f"✓ Connected to Pinecone index: {self.index_name}")
        print(f"✓ Using model: {self.model}")
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context from Pinecone vector database."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract relevant text chunks
            contexts = []
            sources = []
            
            for match in results['matches']:
                if match['score'] > 0.7:  # Only include high-confidence matches
                    contexts.append(match['metadata']['text'])
                    sources.append({
                        'source': match['metadata']['source'],
                        'page': match['metadata'].get('page', 'N/A'),
                        'score': round(match['score'], 3)
                    })
            
            return contexts, sources
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return [], []
    
    def generate_response(self, user_message, contexts):
        """Generate response using OpenRouter with retrieved context."""
        # Prepare system prompt with mental health guidelines
        system_prompt = """You are a compassionate and knowledgeable mental health support chatbot. Your role is to:

1. Provide supportive, empathetic responses based on evidence-based mental health information
2. Use the provided context from mental health resources to inform your answers
3. Always prioritize user safety and well-being
4. Encourage professional help when appropriate (crisis situations, severe symptoms)
5. Never diagnose or replace professional mental health services
6. Be warm, non-judgmental, and validating
7. Provide practical coping strategies when relevant

IMPORTANT SAFETY GUIDELINES:
- If the user mentions suicidal thoughts, self-harm, or crisis: Immediately provide crisis resources (National Suicide Prevention Lifeline: 988, Crisis Text Line: Text HOME to 741741)
- For severe mental health symptoms: Encourage seeking professional help
- You are a support tool, not a replacement for therapy or psychiatric care

Use the following context from mental health resources to inform your response:

{context}

Remember to be compassionate, clear, and helpful while staying within appropriate boundaries."""
        
        # Format context
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        if not contexts:
            context_text = "No specific context available. Use your general mental health knowledge to respond appropriately."
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages for API
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(context=context_text)
            }
        ] + self.conversation_history[-6:]  # Keep last 3 turns for context
        
        try:
            # Call OpenRouter API
            response = requests.post(
                self.openrouter_url,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",  # Optional
                    "X-Title": "Mental Health Chatbot"  # Optional
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            assistant_message = result['choices'][0]['message']['content']
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling OpenRouter API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nDetails: {json.dumps(error_detail, indent=2)}"
                except:
                    error_msg += f"\nResponse: {e.response.text}"
            print(error_msg)
            return "I apologize, but I'm having trouble connecting to my response system. Please try again in a moment."
    
    def chat(self, user_message):
        """Main chat function that combines retrieval and generation."""
        print("\n🔍 Searching knowledge base...")
        
        # Retrieve relevant context
        contexts, sources = self.retrieve_context(user_message)
        
        if contexts:
            print(f"✓ Found {len(contexts)} relevant resources")
        else:
            print("ℹ️  No specific resources found, using general knowledge")
        
        print("\n💭 Generating response...\n")
        
        # Generate response
        response = self.generate_response(user_message, contexts)
        
        return response, sources
    
    def display_sources(self, sources):
        """Display source information for retrieved contexts."""
        if sources:
            print("\n" + "─" * 60)
            print("📚 Sources referenced:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source['source']} (Page {source['page']}) - Relevance: {source['score']}")
            print("─" * 60)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n✓ Conversation history cleared")


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 70)
    print("🌟 Mental Health Support Chatbot".center(70))
    print("=" * 70)
    print("""
Welcome! I'm here to provide supportive mental health information and resources.

⚠️  IMPORTANT REMINDERS:
• I'm a support tool, not a replacement for professional help
• In crisis? Call 988 (Suicide & Crisis Lifeline) or text HOME to 741741
• For emergencies, call 911 or go to your nearest emergency room

I'll do my best to provide helpful, evidence-based information and support.

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
        chatbot = MentalHealthChatbot()
        
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
                print("• I'll search my knowledge base for relevant information")
                print("• I provide supportive responses based on evidence-based resources")
                print("• Type 'clear' to start fresh, 'quit' to exit\n")
                continue
            
            # Process user message
            try:
                response, sources = chatbot.chat(user_input)
                
                # Display response
                print(f"\nChatbot: {response}")
                
                # Display sources
                chatbot.display_sources(sources)
                
                print()  # Extra line for readability
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or rephrase your question.\n")
    
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Take care! 💙")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()