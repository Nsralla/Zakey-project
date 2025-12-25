"""
Quick script to populate the knowledge base with sample AI/ML content.
Run this to get started quickly with your RAG system.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.rag_tool import RAGTool


# Sample knowledge base content about AI frameworks and concepts
SAMPLE_DOCUMENTS = [
    {
        'content': """
        CrewAI Framework Overview:
        CrewAI is a cutting-edge framework designed for orchestrating role-playing autonomous AI agents.
        By enabling agents to assume specific roles, goals, and backstories, CrewAI facilitates 
        collaborative intelligence where multiple agents work together to accomplish complex tasks.
        
        Key Features:
        - Role-based agent design
        - Autonomous task execution
        - Multi-agent collaboration
        - Flexible process flows (sequential, hierarchical)
        - Integration with various LLM providers
        
        Use Cases:
        - Research and analysis workflows
        - Content creation pipelines
        - Data processing automation
        - Customer support systems
        """,
        'metadata': {'source': 'crewai_overview', 'topic': 'frameworks', 'category': 'agents'}
    },
    {
        'content': """
        LangChain Framework:
        LangChain is a framework for developing applications powered by language models. It provides
        tools and abstractions to make it easier to build LLM-powered applications.
        
        Core Components:
        1. Prompts: Templates and management for LLM inputs
        2. Models: Interface to various LLM providers
        3. Chains: Combining multiple LLM calls in sequence
        4. Agents: Dynamic decision-making based on observations
        5. Memory: Maintaining state across interactions
        6. Indexes: Document processing and retrieval
        
        Common Patterns:
        - Question answering over documents
        - Chatbots with memory
        - API interactions
        - Data analysis and summarization
        """,
        'metadata': {'source': 'langchain_guide', 'topic': 'frameworks', 'category': 'llm'}
    },
    {
        'content': """
        Retrieval Augmented Generation (RAG):
        RAG is a technique that enhances language model outputs by retrieving relevant information
        from a knowledge base before generation. This approach reduces hallucination and grounds
        responses in factual data.
        
        RAG Process:
        1. Document Chunking: Break documents into manageable pieces
        2. Embedding: Convert chunks to vector representations
        3. Storage: Store embeddings in a vector database
        4. Retrieval: Search for relevant chunks based on query
        5. Generation: Provide retrieved context to LLM for response
        
        Benefits:
        - Reduced hallucination
        - Up-to-date information
        - Domain-specific knowledge
        - Explainable responses with citations
        
        Popular Vector Databases:
        - ChromaDB, Pinecone, Weaviate, FAISS, Qdrant
        """,
        'metadata': {'source': 'rag_concepts', 'topic': 'techniques', 'category': 'rag'}
    },
    {
        'content': """
        AI Agents and Autonomous Systems:
        An AI agent is a system that perceives its environment, processes information, and takes
        actions to achieve specific goals. Modern AI agents use large language models as their
        reasoning engine.
        
        Agent Components:
        - Perception: Understanding input and context
        - Reasoning: Using LLMs to make decisions
        - Action: Executing tasks using tools
        - Memory: Retaining information across interactions
        
        Agent Types:
        1. Reactive Agents: Respond to immediate stimuli
        2. Deliberative Agents: Plan and reason about actions
        3. Hybrid Agents: Combine reactive and deliberative approaches
        4. Multi-Agent Systems: Multiple agents collaborating
        
        Tools for Agents:
        - Search APIs (web, database)
        - Code execution
        - File operations
        - API integrations
        - Calculator and math operations
        """,
        'metadata': {'source': 'agent_concepts', 'topic': 'agents', 'category': 'theory'}
    },
    {
        'content': """
        Model Context Protocol (MCP):
        MCP is a protocol for connecting AI applications with external data sources and tools.
        It provides a standardized way for AI systems to interact with various services.
        
        Key Concepts:
        - Server: Exposes tools and resources
        - Client: Consumes tools from servers
        - Transport: Communication layer (stdio, HTTP)
        - Tools: Functions that agents can call
        - Resources: Data that can be accessed
        
        Benefits:
        - Standardized integration
        - Reusable components
        - Security and access control
        - Easy scaling
        
        Use Cases:
        - Connecting agents to databases
        - API integrations
        - File system access
        - External service integration
        """,
        'metadata': {'source': 'mcp_protocol', 'topic': 'protocols', 'category': 'integration'}
    },
    {
        'content': """
        Prompt Engineering Best Practices:
        Effective prompt engineering is crucial for getting good results from language models.
        
        Key Principles:
        1. Be specific and clear in instructions
        2. Provide context and examples
        3. Use role-playing and personas
        4. Break complex tasks into steps
        5. Specify output format
        6. Include constraints and requirements
        
        Prompt Patterns:
        - Few-shot learning: Provide examples
        - Chain-of-thought: Ask for step-by-step reasoning
        - Role prompting: Assign an expert role
        - Template-based: Use structured formats
        
        Common Mistakes:
        - Vague or ambiguous instructions
        - Insufficient context
        - Conflicting requirements
        - Not specifying output format
        """,
        'metadata': {'source': 'prompt_engineering', 'topic': 'techniques', 'category': 'prompting'}
    },
    {
        'content': """
        Vector Databases and Embeddings:
        Vector databases store high-dimensional vectors (embeddings) and enable efficient
        similarity search. They are essential for RAG systems and semantic search.
        
        How Embeddings Work:
        - Text is converted to numerical vectors
        - Similar meanings have similar vectors
        - Vector distance measures semantic similarity
        
        Common Embedding Models:
        - OpenAI text-embedding-ada-002
        - Sentence Transformers (open source)
        - Cohere embeddings
        - Google Universal Sentence Encoder
        
        Vector Search:
        - Cosine similarity
        - Euclidean distance
        - Dot product
        
        ChromaDB Features:
        - Simple API
        - Persistent storage
        - Filtering and metadata
        - Multiple distance functions
        """,
        'metadata': {'source': 'vector_databases', 'topic': 'databases', 'category': 'infrastructure'}
    },
    {
        'content': """
        Streamlit for AI Applications:
        Streamlit is a Python framework for building interactive web applications quickly.
        It's particularly popular for data science and ML demos.
        
        Key Features:
        - Pure Python (no HTML/CSS/JS required)
        - Auto-refresh on code changes
        - Built-in widgets (buttons, sliders, inputs)
        - Session state management
        - Caching for performance
        - Easy deployment
        
        Common Components:
        - st.title(), st.header() - Headings
        - st.text_input() - User input
        - st.button() - Buttons
        - st.write() - Display anything
        - st.chat_message() - Chat interfaces
        - st.spinner() - Loading indicators
        
        Best Practices:
        - Use st.cache_data for expensive operations
        - Manage state with session_state
        - Organize code with functions
        - Add error handling
        """,
        'metadata': {'source': 'streamlit_guide', 'topic': 'frontend', 'category': 'ui'}
    },
]


def create_sample_data_directory():
    """Create sample text files in data directory"""
    data_dir = Path("data/knowledge_base")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" Creating sample documents in {data_dir}/\n")
    
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        filename = f"{doc['metadata']['source']}.txt"
        filepath = data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc['content'])
        
        print(f"Created: {filename}")
    
    print(f"\n Created {len(SAMPLE_DOCUMENTS)} sample documents")
    return data_dir


def populate_knowledge_base():
    """Populate the RAG knowledge base"""
    
    print("\n" + "="*60)
    print(" Populating Knowledge Base for AI Agent Project")
    print("="*60 + "\n")
    
    # Step 1: Create sample data files
    data_dir = create_sample_data_directory()
    
    # Step 2: Initialize RAG tool
    print("\n Initializing RAG Tool...")
    rag = RAGTool(
        collection_name="ai_assistant_kb",
        persist_directory="./chroma_db"
    )
    
    # Step 3: Load documents
    print("\n Loading documents into knowledge base...")
    total_chunks = rag.load_directory(
        directory=str(data_dir),
        extensions=['.txt', '.md'],
        chunk_size=500
    )
    
    # Step 4: Show statistics
    print("\n" + "="*60)
    stats = rag.get_collection_stats()
    print(f" Knowledge Base Statistics:")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total Chunks: {stats['total_documents']}")
    print(f"   Storage: {stats['persist_directory']}")
    
    # Step 5: Test searches
    print("\n" + "="*60)
    print(" Testing Knowledge Base with Sample Queries:\n")
    
    test_queries = [
        "What is CrewAI?",
        "How does RAG work?",
        "What are AI agents?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        result = rag.search(query, n_results=2)
        
        if result['success']:
            for i, item in enumerate(result['results'], 1):
                print(f"\n  Result {i} (Relevance: {item['relevance']}):")
                print(f"  Source: {item['metadata'].get('source', 'Unknown')}")
                print(f"  Preview: {item['content'][:150]}...")
        else:
            print(f"   Error: {result['error']}")
    
    print("\n" + "="*60)
    print("Knowledge Base Setup Complete!")
    print("="*60)
    
    print("\n Next Steps:")
    print("   1. Review the data in: ./data/knowledge_base/")
    print("   2. Add your own documents to that folder")
    print("   3. Re-run this script to update the knowledge base")
    print("   4. Use RAGTool in your agents to search this knowledge")
    print("\n See docs/RAG_DATA_RECOMMENDATIONS.md for more ideas")


if __name__ == "__main__":
    try:
        populate_knowledge_base()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

