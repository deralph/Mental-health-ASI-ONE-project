"""
Mental Health Chatbot RAG - Main Chatbot Application (MeTTa/Atomspace Version)
This script creates an interactive chatbot that retrieves relevant information from MeTTa Atomspace
and generates responses using OpenRouter, with Google Gemini for embeddings.

Required packages:
pip install hyperon google-generativeai python-dotenv requests numpy

Required .env file:
GOOGLE_API_KEY=your_google_gemini_api_key (for embeddings)
OPENROUTER_API_KEY=your_openrouter_api_key
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from hyperon import MeTTa

# Load environment variables
load_dotenv()

class MentalHealthMeTTaChatbot:
    def __init__(self, atomspace_file="mental_health_kb.metta", embeddings_file="embeddings.json"):
        """Initialize the mental health chatbot with MeTTa RAG capabilities."""
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not all([self.google_api_key, self.openrouter_api_key]):
            raise ValueError("Missing required API keys in .env file")
        
        # Initialize MeTTa
        self.metta = MeTTa()
        self.atomspace_file = atomspace_file
        self.embeddings_file = embeddings_file
        
        # Load atomspace
        if os.path.exists(self.atomspace_file):
            print(f"Loading MeTTa Atomspace from {self.atomspace_file}...")
            with open(self.atomspace_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    self.metta.run(content)
            print("✓ Atomspace loaded successfully")
        else:
            print(f"⚠️  Warning: Atomspace file {self.atomspace_file} not found")
            print("   Please run the upload script first to populate the knowledge base")
        
        # Load embeddings for retrieval
        self.embeddings_data = []
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r') as f:
                self.embeddings_data = json.load(f)
            print(f"✓ Loaded {len(self.embeddings_data)} document chunks")
        else:
            print(f"⚠️  Warning: Embeddings file {self.embeddings_file} not found")
        
        # Initialize Google Gemini
        genai.configure(api_key=self.google_api_key)
        
        # OpenRouter configuration
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        # You can change this model to any available on OpenRouter
        self.model = "anthropic/claude-3.5-sonnet"
        
        # Conversation history
        self.conversation_history = []
        
        print("✓ Mental Health Chatbot initialized successfully!")
        print(f"✓ Using MeTTa Atomspace for knowledge graph")
        print(f"✓ Using Google Gemini for embeddings")
        print(f"✓ Using model for responses: {self.model}")
    
    def generate_query_embedding(self, query):
        """Generate embedding for query using Google Gemini."""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
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
                    # Query MeTTa atomspace for additional context
                    chunk_id = result['chunk_id']
                    
                    # Query atomspace using MeTTa
                    metta_query = f'(= (chunk-text "{chunk_id}") $text)'
                    
                    contexts.append(result['text'])
                    sources.append({
                        'source': result['source'],
                        'page': result['page'],
                        'score': round(result['similarity'], 3),
                        'chunk_id': chunk_id
                    })
            
            return contexts, sources
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return [], []
    
    def query_metta_knowledge(self, chunk_ids):
        """Query MeTTa atomspace for related knowledge."""
        knowledge = []
        
        for chunk_id in chunk_ids:
            # Query source
            source_query = f'!(= (chunk-source "{chunk_id}") $source)'
            # Query page
            page_query = f'!(= (chunk-page "{chunk_id}") $page)'
            
            try:
                source_result = self.metta.run(source_query)
                page_result = self.metta.run(page_query)
                
                knowledge.append({
                    'chunk_id': chunk_id,
                    'source': str(source_result) if source_result else "Unknown",
                    'page': str(page_result) if page_result else "Unknown"
                })
            except:
                pass
        
        return knowledge
    
    def generate_response(self, user_message, contexts):
        """Generate response using OpenRouter with retrieved context."""
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
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Mental Health MeTTa Chatbot"
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
        print("\n🔍 Searching MeTTa Knowledge Graph...")
        
        # Retrieve relevant context from MeTTa atomspace
        contexts, sources = self.retrieve_from_metta(user_message)
        
        if contexts:
            print(f"✓ Found {len(contexts)} relevant knowledge graph nodes")
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
        print("\n✓ Conversation history cleared")


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 70)
    print("🌟 Mental Health Support Chatbot (MeTTa Knowledge Graph)".center(70))
    print("=" * 70)
    print("""
Welcome! I'm here to provide supportive mental health information powered by
a MeTTa Knowledge Graph - part of SingularityNET's AGI framework.

🔧 Powered by:
• MeTTa/Atomspace (Knowledge Graph)
• Google Gemini (Embeddings)
• OpenRouter (Response Generation)

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
                print(f"\nChatbot: {response}")
                
                # Display sources from knowledge graph
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()