"""
Test script for LLM-enhanced RAG search.

This demonstrates how the RAG tool now synthesizes answers using an LLM
before returning results to the user.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.rag_tool import RAGTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_rag_with_llm():
    """Test RAG with LLM synthesis"""
    
    print("=" * 70)
    print("Testing LLM-Enhanced RAG Search")
    print("=" * 70)
    print()
    
    # Initialize RAG
    print("1. Initializing RAG Tool...")
    rag = RAGTool(collection_name="ai_assistant_kb")
    
    # Check if knowledge base has content
    stats = rag.get_collection_stats()
    print(f"   Knowledge base: {stats.get('total_documents', 0)} documents")
    print()
    
    if stats.get('total_documents', 0) == 0:
        print("❌ Knowledge base is empty!")
        print("   Run: python scripts/populate_knowledge_base.py")
        return
    
    # Test queries
    test_queries = [
        "What is CrewAI?",
        "How does RAG work?",
        "What are AI agents?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Testing query: '{query}'")
        print("-" * 70)
        
        # Search with LLM synthesis
        result = rag.search_with_llm(query, n_results=3)
        
        if result['success']:
            print("✅ Success!")
            print()
            print("AI-Synthesized Answer:")
            print("-" * 70)
            print(result['answer'])
            print()
            
            if result.get('sources'):
                print("Sources Used:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source['source']} (Relevance: {source['relevance']:.1%})")
                print()
            
            if result.get('note'):
                print(f"Note: {result['note']}")
                print()
        else:
            print(f"❌ Failed: {result.get('error')}")
            print()
        
        print("=" * 70)
        print()


def compare_raw_vs_llm():
    """Compare raw retrieval vs LLM-enhanced results"""
    
    print("=" * 70)
    print("Comparison: Raw Retrieval vs LLM-Enhanced")
    print("=" * 70)
    print()
    
    rag = RAGTool(collection_name="ai_assistant_kb")
    
    query = "What is CrewAI?"
    print(f"Query: {query}")
    print()
    
    # Raw retrieval
    print("--- RAW RETRIEVAL (No LLM) ---")
    print("-" * 70)
    raw_result = rag.search(query, n_results=3)
    if raw_result['success'] and raw_result['results']:
        for i, item in enumerate(raw_result['results'], 1):
            print(f"{i}. {item['content'][:200]}...")
            print(f"   Source: {item['metadata'].get('source', 'Unknown')}")
            print()
    print()
    
    # LLM-enhanced
    print("--- LLM-ENHANCED (Synthesized Answer) ---")
    print("-" * 70)
    llm_result = rag.search_with_llm(query, n_results=3)
    if llm_result['success']:
        print(llm_result['answer'])
        print()
    
    print("=" * 70)
    print()
    print("📊 Comparison:")
    print("  • Raw retrieval: Shows document chunks directly")
    print("  • LLM-enhanced: Synthesizes info into coherent answer")
    print("  • LLM version cites sources and provides context")
    print()


if __name__ == "__main__":
    print()
    print("🔬 ZAKEY - LLM-Enhanced RAG Test")
    print()
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: No API key found (OPENROUTER_API_KEY or OPENAI_API_KEY)")
        print("   LLM synthesis will be skipped, raw results will be shown")
        print()
    
    try:
        # Run tests
        test_rag_with_llm()
        
        # Compare
        compare_raw_vs_llm()
        
        print("✅ All tests completed!")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

