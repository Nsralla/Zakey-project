"""
Quick test to verify all packages are installed correctly
"""
import sys

def test_imports():
    """Test if all critical packages can be imported"""
    
    print("🧪 Testing package imports...\n")
    
    tests = [
        ("CrewAI", "crewai"),
        ("LangChain", "langchain"),
        ("OpenAI", "openai"),
        ("Streamlit", "streamlit"),
        ("Tavily", "tavily"),
        ("ChromaDB", "chromadb"),
        ("Sentence Transformers", "sentence_transformers"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Python-Dotenv", "dotenv"),
        ("BeautifulSoup4", "bs4"),
        ("Requests", "requests"),
    ]
    
    passed = 0
    failed = []
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed.append(name)
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} packages loaded successfully")
    
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return False
    else:
        print("✨ All packages are ready to use!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

