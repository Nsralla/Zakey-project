"""
RAG Tool for AI Agents
Provides document retrieval capabilities using ChromaDB and embeddings.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGTool:
    """
    RAG (Retrieval Augmented Generation) tool for document-based question answering.
    
    Features:
    - Document ingestion (PDF, TXT, MD, etc.)
    - Vector storage with ChromaDB
    - Semantic search and retrieval
    - Context-aware responses
    """
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers"
    ):
        """
        Initialize the RAG Tool.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ("sentence-transformers" or "openai")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_type = embedding_model
        
        # Initialize ChromaDB
        self._init_chroma()
        
        # Initialize embeddings
        self._init_embeddings()
        
        logger.info(f"RAG Tool initialized with collection: {collection_name}")
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize Chroma client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document knowledge base for RAG"}
            )
            
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
            
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install it with: pip install chromadb"
            )
    
    def _init_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_model_type == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                logger.info("Using OpenAI embeddings")
            except ImportError:
                raise ImportError("langchain-openai not installed")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Using Sentence Transformers embeddings")
            except ImportError:
                raise ImportError("sentence-transformers not installed")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        if self.embedding_model_type == "openai":
            return self.embeddings.embed_query(text)
        else:
            return self.embeddings.encode(text).tolist()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents provided")
            return 0
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        try:
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def load_text_file(self, file_path: str, chunk_size: int = 500) -> int:
        """
        Load and chunk a text file into the knowledge base.
        
        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks (in characters)
            
        Returns:
            Number of chunks added
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = self._chunk_text(content, chunk_size)
            
            # Create metadata
            metadatas = [
                {
                    "source": os.path.basename(file_path),
                    "chunk_id": i,
                    "file_path": file_path
                }
                for i in range(len(chunks))
            ]
            
            # Create IDs
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ids = [f"{base_name}_chunk_{i}" for i in range(len(chunks))]
            
            return self.add_documents(chunks, metadatas, ids)
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return 0
    
    def load_directory(
        self,
        directory: str,
        extensions: List[str] = ['.txt', '.md', '.py'],
        chunk_size: int = 500
    ) -> int:
        """
        Load all files from a directory into the knowledge base.
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to load
            chunk_size: Size of text chunks
            
        Returns:
            Total number of chunks added
        """
        total_chunks = 0
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory}")
            return 0
        
        for ext in extensions:
            for file_path in directory_path.rglob(f"*{ext}"):
                logger.info(f"Loading: {file_path}")
                chunks = self.load_text_file(str(file_path), chunk_size)
                total_chunks += chunks
        
        logger.info(f"Loaded {total_chunks} chunks from {directory}")
        return total_chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # At least 50% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        min_relevance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            Dictionary with search results
        """
        if not query or not query.strip():
            return {
                'success': False,
                'error': 'Query cannot be empty',
                'results': []
            }
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results.get('distances') else 0
                    # Convert distance to similarity score (inverse relationship)
                    relevance = 1 - min(distance, 1.0)
                    
                    if relevance >= min_relevance:
                        formatted_results.append({
                            'content': doc,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                            'relevance': round(relevance, 3),
                            'id': results['ids'][0][i] if results.get('ids') else None
                        })
            
            logger.info(f"Found {len(formatted_results)} results for: {query}")
            
            return {
                'success': True,
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    def get_context(self, query: str, n_results: int = 3) -> str:
        """
        Get relevant context as a single string (useful for LLM prompts).
        
        Args:
            query: Search query
            n_results: Number of results to include
            
        Returns:
            Concatenated context string
        """
        result = self.search(query, n_results=n_results)
        
        if not result['success'] or not result['results']:
            return ""
        
        context_parts = []
        for i, item in enumerate(result['results'], 1):
            source = item['metadata'].get('source', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{item['content']}\n")
        
        return "\n".join(context_parts)
    
    def search_with_llm(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search knowledge base and synthesize results using LLM.
        
        Args:
            query: User's question
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary with LLM-synthesized answer and sources
        """
        # Retrieve relevant documents
        search_result = self.search(query, n_results=n_results)
        
        if not search_result['success'] or not search_result['results']:
            return {
                'success': False,
                'error': 'No relevant documents found',
                'answer': 'No information found in the knowledge base.',
                'sources': []
            }
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, item in enumerate(search_result['results'], 1):
            source = item['metadata'].get('source', 'Unknown')
            context_parts.append(f"Document {i} (Source: {source}):\n{item['content']}\n")
            sources.append({
                'source': source,
                'relevance': item['relevance'],
                'content_preview': item['content'][:200] + '...'
            })
        
        context = "\n".join(context_parts)
        
        # Synthesize with LLM
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Get API configuration
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("MODEL", "google/gemini-2.0-flash-exp:free")
            
            if not api_key:
                # Return raw context if no API key
                return {
                    'success': True,
                    'answer': f"**Retrieved Information:**\n\n{context}",
                    'sources': sources,
                    'note': 'LLM synthesis unavailable (no API key). Showing raw results.'
                }
            
            # Initialize LLM
            if "OPENROUTER" in os.environ or model_name.count('/') >= 1:
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    max_tokens=800,
                    temperature=0.3
                )
            else:
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    max_tokens=800,
                    temperature=0.3
                )
            
            # Create synthesis prompt
            prompt = f"""Based on the following documents from the knowledge base, provide a comprehensive and accurate answer to the user's question.

User Question: {query}

Retrieved Documents:
{context}

Instructions:
1. Answer the question directly and clearly
2. Use information ONLY from the provided documents
3. Cite sources when referencing specific information (e.g., "According to [Source]...")
4. If the documents don't fully answer the question, say so
5. Keep the response well-structured and professional

Answer:"""
            
            # Get LLM response
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # Fallback to raw context
            return {
                'success': True,
                'answer': f"**Retrieved Information:**\n\n{context}",
                'sources': sources,
                'note': f'LLM synthesis failed: {str(e)}. Showing raw results.'
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")


# Convenience function
def search_documents(query: str, rag_tool: RAGTool = None) -> str:
    """
    Convenience function for document search.
    
    Args:
        query: Search query
        rag_tool: Existing RAG tool instance (creates new one if None)
        
    Returns:
        Context string
    """
    if rag_tool is None:
        rag_tool = RAGTool()
    
    return rag_tool.get_context(query)


# Example usage
if __name__ == "__main__":
    print("Testing RAG Tool\n")
    
    # Initialize RAG
    rag = RAGTool(collection_name="test_kb")
    
    # Example 1: Add sample documents
    print("Test 1: Adding sample documents")
    print("-" * 50)
    
    sample_docs = [
        "CrewAI is a framework for orchestrating role-playing autonomous AI agents. It enables agents to work together to accomplish complex tasks.",
        "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents.",
        "Tavily is a search API optimized for LLMs and RAG systems. It provides fast, reliable web search results.",
        "Vector databases store embeddings and enable semantic search. ChromaDB is a popular open-source vector database.",
    ]
    
    metadatas = [
        {"source": "crewai_docs", "topic": "frameworks"},
        {"source": "langchain_docs", "topic": "frameworks"},
        {"source": "tavily_docs", "topic": "search"},
        {"source": "vector_db_docs", "topic": "databases"},
    ]
    
    added = rag.add_documents(sample_docs, metadatas)
    print(f"Added {added} documents\n")
    
    # Example 2: Search
    print("Test 2: Searching knowledge base")
    print("-" * 50)
    
    query = "What is CrewAI?"
    result = rag.search(query, n_results=2)
    
    if result['success']:
        print(f"Query: {result['query']}")
        print(f"Found {result['total_results']} results:\n")
        
        for i, item in enumerate(result['results'], 1):
            print(f"{i}. Relevance: {item['relevance']}")
            print(f"   Source: {item['metadata'].get('source', 'Unknown')}")
            print(f"   Content: {item['content'][:100]}...\n")
    
    # Example 3: Get context for LLM
    print("Test 3: Get context string")
    print("-" * 50)
    
    context = rag.get_context("vector databases", n_results=2)
    print(f"Context:\n{context[:200]}...\n")
    
    # Stats
    print("=" * 50)
    stats = rag.get_collection_stats()
    print(f"Collection stats: {stats}")
    print("\nAll tests completed!")

