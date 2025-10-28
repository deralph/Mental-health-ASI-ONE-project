"""
Mental Health Chatbot RAG - Document Upload Script (MeTTa/Atomspace Version)
This script processes PDF documents and uploads them to MeTTa Atomspace knowledge graph.

Required packages:
pip install hyperon pypdf2 google-generativeai python-dotenv langchain langchain-community

Required .env file:
GOOGLE_API_KEY=your_google_gemini_api_key
"""

import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
import hashlib
from hyperon import MeTTa, E, S, V, ValueAtom

# Load environment variables
load_dotenv()

class MentalHealthMeTTaUploader:
    def __init__(self, atomspace_file="mental_health_kb.metta"):
        """Initialize the document uploader with MeTTa Atomspace."""
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in .env file")
        
        # Initialize Google Gemini
        genai.configure(api_key=self.google_api_key)
        
        # Initialize MeTTa
        self.metta = MeTTa()
        self.atomspace_file = atomspace_file
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Load existing atomspace if exists
        if os.path.exists(self.atomspace_file):
            print(f"Loading existing atomspace from {self.atomspace_file}")
            with open(self.atomspace_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    self.metta.run(content)
        
        print(f"✓ MeTTa Atomspace initialized")
        print(f"✓ Knowledge base file: {self.atomspace_file}")
    
    def generate_embedding(self, text):
        """Generate embedding using Google Gemini."""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    def load_pdf(self, pdf_path):
        """Load and extract text from PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"\nLoading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        return documents
    
    def add_to_atomspace(self, chunk_id, text, embedding, source, page, chunk_index):
        """Add document chunk to MeTTa Atomspace as knowledge graph nodes."""
        
        # Convert embedding list to string representation for MeTTa
        embedding_str = str(embedding[:10])  # Store first 10 dims as sample
        
        # Escape special characters in text
        text_escaped = text.replace('"', '\\"').replace('\n', ' ')
        
        # Create MeTTa expressions for knowledge representation
        # Structure: (DocumentChunk chunk_id text source page embedding_sample)
        metta_expr = f'''
; Document Chunk {chunk_id}
(= (chunk-text "{chunk_id}") "{text_escaped[:500]}")
(= (chunk-source "{chunk_id}") "{source}")
(= (chunk-page "{chunk_id}") {page})
(= (chunk-index "{chunk_id}") {chunk_index})
(= (chunk-embedding-sample "{chunk_id}") {embedding_str})

; Mental Health Knowledge
(MentalHealthDocument "{chunk_id}" "{source}" {page})
'''
        
        # Execute MeTTa expression to add to atomspace
        self.metta.run(metta_expr)
        
        return metta_expr
    
    def process_documents(self, documents, source_name):
        """Split documents into chunks and prepare for upload."""
        print("\nSplitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        metta_expressions = []
        embeddings_data = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID
            chunk_id = hashlib.md5(
                f"{source_name}_{i}_{chunk.page_content[:50]}".encode()
            ).hexdigest()[:16]
            
            # Generate embedding
            print(f"Processing chunk {i+1}/{len(chunks)}...", end='\r')
            embedding = self.generate_embedding(chunk.page_content)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            
            # Add to atomspace
            metta_expr = self.add_to_atomspace(
                chunk_id=chunk_id,
                text=chunk.page_content,
                embedding=embedding,
                source=source_name,
                page=chunk.metadata.get("page", 0),
                chunk_index=i
            )
            
            metta_expressions.append(metta_expr)
            
            # Store full embedding separately for retrieval
            embeddings_data.append({
                'chunk_id': chunk_id,
                'text': chunk.page_content,
                'embedding': embedding,
                'source': source_name,
                'page': chunk.metadata.get("page", 0)
            })
        
        print(f"\nAdded {len(chunks)} chunks to MeTTa Atomspace")
        return metta_expressions, embeddings_data
    
    def save_atomspace(self, metta_expressions):
        """Save atomspace to file."""
        with open(self.atomspace_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(metta_expressions))
            f.write('\n')
        print(f"✓ Saved atomspace to {self.atomspace_file}")
    
    def save_embeddings(self, embeddings_data, embeddings_file="embeddings.json"):
        """Save embeddings separately for efficient retrieval."""
        import json
        
        existing_data = []
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r') as f:
                existing_data = json.load(f)
        
        existing_data.extend(embeddings_data)
        
        with open(embeddings_file, 'w') as f:
            json.dump(existing_data, f)
        
        print(f"✓ Saved {len(embeddings_data)} embeddings to {embeddings_file}")
    
    def upload_pdf(self, pdf_path, source_name=None):
        """Complete pipeline: load, process, and upload PDF."""
        if source_name is None:
            source_name = os.path.basename(pdf_path)
        
        try:
            # Load PDF
            documents = self.load_pdf(pdf_path)
            
            # Process and create MeTTa expressions
            metta_expressions, embeddings_data = self.process_documents(documents, source_name)
            
            # Save to atomspace file
            self.save_atomspace(metta_expressions)
            
            # Save embeddings for retrieval
            self.save_embeddings(embeddings_data)
            
            print(f"\n✓ Successfully uploaded '{source_name}' to MeTTa Atomspace!")
            return True
            
        except Exception as e:
            print(f"\n✗ Error uploading PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def query_atomspace(self, query):
        """Query the atomspace using MeTTa."""
        result = self.metta.run(query)
        return result


def main():
    """Main function to upload mental health documents."""
    print("=" * 70)
    print("Mental Health Chatbot - MeTTa Knowledge Graph Upload System")
    print("Using: Google Gemini Embeddings + MeTTa Atomspace")
    print("=" * 70)
    
    # Initialize uploader
    try:
        uploader = MentalHealthMeTTaUploader()
    except Exception as e:
        print(f"\n✗ Failed to initialize uploader: {str(e)}")
        print("\nPlease check your .env file and ensure you have:")
        print("  - GOOGLE_API_KEY")
        print("\nAlso ensure you have installed hyperon:")
        print("  pip install hyperon")
        return
    
    print("\n📚 Suggested Mental Health PDFs to download and use:")
    print("1. WHO Mental Health Guide: https://www.who.int/publications")
    print("2. NIMH Publications: https://www.nimh.nih.gov/health/publications")
    print("3. CBT Self-Help Resources")
    print("4. Anxiety and Depression Guides")
    print("5. Mindfulness and Meditation Resources")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Upload Mode")
    print("=" * 70)
    
    while True:
        pdf_path = input("\nEnter PDF path (or 'quit' to exit): ").strip()
        
        if pdf_path.lower() == 'quit':
            break
        
        if not pdf_path:
            continue
        
        # Remove quotes if user wrapped path in quotes
        pdf_path = pdf_path.strip('"').strip("'")
        
        source_name = input("Enter source name (or press Enter to use filename): ").strip()
        
        if not source_name:
            source_name = os.path.basename(pdf_path)
        
        uploader.upload_pdf(pdf_path, source_name)
        
        another = input("\nUpload another file? (y/n): ").strip().lower()
        if another != 'y':
            break
    
    print("\n" + "=" * 70)
    print("Upload session complete!")
    print(f"Knowledge base saved to: {uploader.atomspace_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()