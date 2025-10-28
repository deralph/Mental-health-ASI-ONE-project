"""
Mental Health Chatbot RAG - Main Chatbot Application (OpenAI + MeTTa Version)
This script creates an interactive chatbot that retrieves relevant information from MeTTa Atomspace
and generates responses using OpenAI, with OpenAI embeddings.

Required packages:
pip install hyperon openai python-dotenv numpy

Required .env file:
OPENAI_API_KEY=your_openai_api_key
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from hyperon import MeTTa

# Load environment variables
load_dotenv()

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

