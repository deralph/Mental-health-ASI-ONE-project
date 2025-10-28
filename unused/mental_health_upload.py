"""
Mental Health Chatbot RAG - Document Upload Script
This script processes PDF documents and uploads them to Pinecone vector database.

Required packages:
pip install pinecone-client pypdf2 openai python-dotenv langchain langchain-community langchain-openai tiktoken

Required .env file:
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
OPENAI_API_KEY=your_openai_api_key (for embeddings)
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import hashlib

# Load environment variables
load_dotenv()

class MentalHealthDocumentUploader:
    def __init__(self):
        """Initialize the document uploader with Pinecone and embeddings."""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("Missing required API keys in .env file")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "mental-health-chatbot"
        self.dimension = 1536  # OpenAI embedding dimension
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self._setup_index()
    
    def _setup_index(self):
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index: {self.index_name}")
        print(f"Index stats: {self.index.describe_index_stats()}")
    
    def load_pdf(self, pdf_path):
        """Load and extract text from PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"\nLoading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        return documents
    
    def process_documents(self, documents, source_name):
        """Split documents into chunks and prepare for upload."""
        print("\nSplitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Prepare vectors for upload
        vectors = []
        for i, chunk in enumerate(chunks):
            # Create unique ID
            chunk_id = hashlib.md5(
                f"{source_name}_{i}_{chunk.page_content[:50]}".encode()
            ).hexdigest()
            
            # Generate embedding
            print(f"Processing chunk {i+1}/{len(chunks)}...", end='\r')
            embedding = self.embeddings.embed_query(chunk.page_content)
            
            # Prepare metadata
            metadata = {
                "text": chunk.page_content,
                "source": source_name,
                "page": chunk.metadata.get("page", 0),
                "chunk_id": i
            }
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })
        
        print(f"\nPrepared {len(vectors)} vectors for upload")
        return vectors
    
    def upload_to_pinecone(self, vectors, batch_size=100):
        """Upload vectors to Pinecone in batches."""
        print(f"\nUploading {len(vectors)} vectors to Pinecone...")
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        
        print("✓ Upload complete!")
        
        # Wait a moment for indexing
        time.sleep(2)
        stats = self.index.describe_index_stats()
        print(f"\nUpdated index stats: {stats}")
    
    def upload_pdf(self, pdf_path, source_name=None):
        """Complete pipeline: load, process, and upload PDF."""
        if source_name is None:
            source_name = os.path.basename(pdf_path)
        
        try:
            # Load PDF
            documents = self.load_pdf(pdf_path)
            
            # Process and create vectors
            vectors = self.process_documents(documents, source_name)
            
            # Upload to Pinecone
            self.upload_to_pinecone(vectors)
            
            print(f"\n✓ Successfully uploaded '{source_name}' to Pinecone!")
            return True
            
        except Exception as e:
            print(f"\n✗ Error uploading PDF: {str(e)}")
            return False


def main():
    """Main function to upload mental health documents."""
    print("=" * 60)
    print("Mental Health Chatbot - Document Upload System")
    print("=" * 60)
    
    # Initialize uploader
    uploader = MentalHealthDocumentUploader()
    
    # Example: Upload mental health PDFs
    # Replace these with your actual PDF paths
    pdf_files = [
        # Example paths - replace with your actual files
        "mental_health_guide.pdf",
        "cbt_workbook.pdf",
        "anxiety_management.pdf",
        "mindfulness_guide.pdf"
    ]
    
    print("\nSuggested PDFs to download and use:")
    print("1. WHO Mental Health Guide: https://www.who.int/publications")
    print("2. CBT Self-Help Guide: Available from various mental health organizations")
    print("3. Anxiety and Depression Guide: Check ADAA or NIMH websites")
    print("4. Mindfulness Resources: Free resources from mindfulness organizations")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Upload Mode")
    print("=" * 60)
    
    while True:
        pdf_path = input("\nEnter PDF path (or 'quit' to exit): ").strip()
        
        if pdf_path.lower() == 'quit':
            break
        
        if not pdf_path:
            continue
        
        source_name = input("Enter source name (or press Enter to use filename): ").strip()
        
        if not source_name:
            source_name = os.path.basename(pdf_path)
        
        uploader.upload_pdf(pdf_path, source_name)
        
        another = input("\nUpload another file? (y/n): ").strip().lower()
        if another != 'y':
            break
    
    print("\n" + "=" * 60)
    print("Upload session complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()