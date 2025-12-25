"""
MCP Client Example
Shows how to use the MCP server endpoints from Python.
"""

import requests
import json
from typing import Dict, Any

# Server base URL
BASE_URL = "http://localhost:8000"


class MCPClient:
    """Client for interacting with the AI Agent MCP Server"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check if server is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web"""
        response = requests.post(
            f"{self.base_url}/tools/web-search",
            json={"query": query, "max_results": max_results}
        )
        return response.json()
    
    def kb_search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Search knowledge base"""
        response = requests.post(
            f"{self.base_url}/tools/kb-search",
            json={"query": query, "n_results": n_results}
        )
        return response.json()
    
    def research_topic(self, topic: str, save_report: bool = True) -> Dict[str, Any]:
        """Run AI agents to research a topic"""
        response = requests.post(
            f"{self.base_url}/agents/research",
            json={"topic": topic, "save_report": save_report}
        )
        return response.json()
    
    def summarize(self, text: str, max_sentences: int = 5) -> Dict[str, Any]:
        """Summarize text"""
        response = requests.post(
            f"{self.base_url}/tools/summarize",
            json={"text": text, "max_sentences": max_sentences}
        )
        return response.json()
    
    def get_kb_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        response = requests.get(f"{self.base_url}/kb/stats")
        return response.json()
    
    def list_tools(self) -> Dict[str, Any]:
        """List all available MCP tools"""
        response = requests.get(f"{self.base_url}/mcp/tools")
        return response.json()


def example_usage():
    """Examples of using the MCP client"""
    
    print("\n" + "="*70)
    print("MCP Client Examples")
    print("="*70 + "\n")
    
    # Initialize client
    client = MCPClient()
    
    # Example 1: Health check
    print("1. Health Check")
    print("-" * 70)
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Tools available: {health['tools']}\n")
    
    # Example 2: Search knowledge base
    print("2. Knowledge Base Search")
    print("-" * 70)
    kb_result = client.kb_search("What is CrewAI?")
    print(f"Query: {kb_result['query']}")
    print(f"Results found: {kb_result['total_results']}")
    if kb_result['results']:
        print(f"First result: {kb_result['results'][0]['content'][:200]}...\n")
    
    # Example 3: Web search
    print("3. Web Search")
    print("-" * 70)
    try:
        web_result = client.web_search("AI agents latest news")
        print(f"Query: {web_result['query']}")
        print(f"Results: {len(web_result['results'])}")
        if web_result.get('answer'):
            print(f"Answer: {web_result['answer'][:200]}...\n")
    except Exception as e:
        print(f"Web search not available: {e}\n")
    
    # Example 4: Summarize text
    print("4. Text Summarization")
    print("-" * 70)
    sample_text = """
    Artificial Intelligence agents are autonomous systems that can perceive their
    environment, process information, make decisions, and take actions to achieve
    specific goals. Modern AI agents use large language models as their reasoning
    engine, allowing them to understand complex instructions and adapt to new situations.
    They can be equipped with various tools to interact with external systems, search
    databases, or perform calculations.
    """
    summary_result = client.summarize(sample_text, max_sentences=2)
    print(f"Original length: {summary_result['original_length']} chars")
    print(f"Summary length: {summary_result['summary_length']} chars")
    print(f"Summary: {summary_result['summary']}\n")
    
    # Example 5: Research with agents (commented out - takes longer)
    print("5. AI Agent Research (commented out - uncomment to run)")
    print("-" * 70)
    print("# Uncomment to run:")
    print("# research_result = client.research_topic('AI agent frameworks')")
    print("# print(research_result['result'][:500])")
    print()
    
    # Example 6: List available tools
    print("6. Available MCP Tools")
    print("-" * 70)
    tools = client.list_tools()
    for tool in tools['tools']:
        print(f"- {tool['name']}: {tool['description']}")
    print()
    
    print("="*70)
    print("Examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        example_usage()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to MCP server")
        print("Make sure the server is running:")
        print("  python mcp_server/server.py")
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

