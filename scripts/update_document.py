"""
Quick script to update a single document in the knowledge base.
Run this after editing a file in data/knowledge_base/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.rag_tool import RAGTool

def update_document(filename: str):
    """Update a single document in the knowledge base"""
    
    # Initialize RAG tool
    rag = RAGTool(collection_name="ai_assistant_kb")
    
    # Path to the document
    doc_path = Path(__file__).parent.parent / "data" / "knowledge_base" / filename
    
    if not doc_path.exists():
        print(f" File not found: {doc_path}")
        return
    
    print(f" Updating document: {filename}")
    
    # Read the file
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove old chunks for this file (if any)
    # ChromaDB doesn't have a direct "update" - we need to delete and re-add
    try:
        # Get all documents
        results = rag.collection.get()
        
        # Find IDs of chunks from this file
        old_ids = []
        for i, metadata in enumerate(results['metadatas']):
            if metadata.get('source') == filename:
                old_ids.append(results['ids'][i])
        
        # Delete old chunks
        if old_ids:
            rag.collection.delete(ids=old_ids)
            print(f" Removed {len(old_ids)} old chunks")
    except Exception as e:
        print(f"  Could not remove old chunks: {e}")
    
    # Add updated document
    chunks_added = rag.load_text_file(str(doc_path), chunk_size=500)
    
    print(f" Added {chunks_added} new chunks from {filename}")
    print(f" Total documents in KB: {rag.get_collection_stats()['total_documents']}")
    
    # Test query
    print(f"\n Testing search...")
    test_result = rag.search(f"information from {filename}", n_results=1)
    if test_result['success'] and test_result['results']:
        print(f" Search working! Found content from {filename}")
    else:
        print(f"  Search returned no results")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default to langchain_guide.txt
        filename = "langchain_guide.txt"
    
    update_document(filename)

