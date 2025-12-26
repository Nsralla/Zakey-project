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


def populate_knowledge_base():
    """Populate the RAG knowledge base"""
    
    print("\n" + "="*60)
    print(" Populating Knowledge Base for AI Agent Project")
    print("="*60 + "\n")
    
    # Step 1: Create sample data files
    data_dir =  Path("data/knowledge_base")
    
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

