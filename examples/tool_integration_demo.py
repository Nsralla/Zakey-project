"""
Example: Integrating All Tools with CrewAI Agents
Demonstrates how to use search_tool, rag_tool, and custom_tools together.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Import tools
from tools.search_tool import TavilySearchTool
from tools.rag_tool import RAGTool
from tools.custom_tools import FileWriterTool, SummaryTool, DataParserTool


def demo_tool_integration():
    """Demonstrate using all tools in a workflow"""
    
    print("\n" + "="*70)
    print("AI Agent Tools Integration Demo")
    print("="*70)
    
    # Initialize all tools
    print("\nInitializing tools...")
    
    # Check for API keys
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not set. Tavily searches will fail.")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some features may be limited.")
    
    # Initialize tools
    tavily = TavilySearchTool() if os.getenv("TAVILY_API_KEY") else None
    rag = RAGTool(collection_name="ai_assistant_kb")
    writer = FileWriterTool(output_dir="./output/demo")
    summarizer = SummaryTool(use_llm=True)
    parser = DataParserTool()
    
    print("Tools initialized successfully!\n")
    
    # =====================================================================
    # Workflow 1: Web Research + Summary + Save
    # =====================================================================
    print("\n" + "="*70)
    print("Workflow 1: Web Research → Summarize → Save")
    print("="*70)
    
    query = "What is CrewAI framework?"
    print(f"\nQuery: {query}")
    
    if tavily:
        # Step 1: Search web with Tavily
        print("\nStep 1: Searching web...")
        web_results = tavily.search(query, max_results=3)
        
        if web_results['success']:
            print(f"Found {len(web_results['results'])} results")
            
            # Combine results into text
            combined_text = ""
            if 'answer' in web_results:
                combined_text += f"Answer: {web_results['answer']}\n\n"
            
            for i, result in enumerate(web_results['results'], 1):
                combined_text += f"\n{i}. {result['title']}\n"
                combined_text += f"{result['snippet']}\n"
                combined_text += f"Source: {result['url']}\n"
            
            # Step 2: Summarize results
            print("\nStep 2: Creating summary...")
            summary_result = summarizer.create_summary(combined_text, max_sentences=3)
            
            if summary_result['success']:
                print(f"Summary: {summary_result['summary'][:200]}...")
                
                # Step 3: Save to file
                print("\nStep 3: Saving results...")
                
                # Save as markdown
                markdown_content = f"# Research Results: {query}\n\n"
                markdown_content += f"## Summary\n{summary_result['summary']}\n\n"
                markdown_content += f"## Detailed Results\n{combined_text}\n"
                
                save_result = writer.write_markdown(
                    markdown_content,
                    "web_research_results",
                    title="Web Research Results"
                )
                
                if save_result['success']:
                    print(f"Saved to: {save_result['file_path']}")
                
                # Save as JSON
                json_data = {
                    'query': query,
                    'summary': summary_result['summary'],
                    'results': web_results['results'],
                    'timestamp': str(Path(save_result['file_path']).stat().st_mtime)
                }
                json_result = writer.write_json(json_data, "web_research_data")
                print(f"JSON saved to: {json_result['file_path']}")
    else:
        print("Skipping web search (Tavily API key not set)")
    
    # =====================================================================
    # Workflow 2: RAG Search + Extract Data + Save
    # =====================================================================
    print("\n" + "="*70)
    print("Workflow 2: RAG Search → Extract Data → Save")
    print("="*70)
    
    rag_query = "What are AI agents?"
    print(f"\nQuery: {rag_query}")
    
    # Step 1: Search knowledge base
    print("\nStep 1: Searching knowledge base...")
    rag_results = rag.search(rag_query, n_results=3)
    
    if rag_results['success'] and rag_results['results']:
        print(f"Found {len(rag_results['results'])} results in knowledge base")
        
        # Combine RAG results
        rag_text = ""
        for i, result in enumerate(rag_results['results'], 1):
            rag_text += f"\n{i}. {result['content']}\n"
            rag_text += f"Source: {result['metadata'].get('source', 'Unknown')}\n"
            rag_text += f"Relevance: {result['relevance']}\n"
        
        # Step 2: Extract key information
        print("\nStep 2: Extracting key points...")
        key_points = summarizer.extract_key_points(rag_text, num_points=5)
        
        if key_points['success']:
            print("Key points:")
            for i, point in enumerate(key_points['key_points'], 1):
                print(f"  {i}. {point[:80]}...")
        
        # Extract structured data
        print("\nStep 3: Parsing structured data...")
        urls = parser.extract_urls(rag_text)
        if urls:
            print(f"URLs found: {urls}")
        
        # Step 4: Save results
        print("\nStep 4: Saving RAG results...")
        
        # Save as text
        text_content = f"RAG Search Results for: {rag_query}\n\n"
        text_content += f"Key Points:\n"
        for i, point in enumerate(key_points['key_points'], 1):
            text_content += f"{i}. {point}\n"
        text_content += f"\n\nDetailed Results:\n{rag_text}"
        
        save_result = writer.write_text(text_content, "rag_search_results")
        print(f"Saved to: {save_result['file_path']}")
        
        # Save as structured CSV
        csv_data = []
        for result in rag_results['results']:
            csv_data.append({
                'content_preview': result['content'][:100] + "...",
                'source': result['metadata'].get('source', 'Unknown'),
                'relevance': result['relevance']
            })
        
        csv_result = writer.write_csv(csv_data, "rag_results")
        print(f"CSV saved to: {csv_result['file_path']}")
    
    else:
        print("No results found in knowledge base.")
        print("Run 'python scripts/populate_knowledge_base.py' first!")
    
    # =====================================================================
    # Workflow 3: Combined Search (RAG + Web) + Analysis
    # =====================================================================
    print("\n" + "="*70)
    print("Workflow 3: Combined Search (RAG + Web) → Analyze → Report")
    print("="*70)
    
    combined_query = "How to create AI agents?"
    print(f"\nQuery: {combined_query}")
    
    all_results = []
    
    # Search RAG first
    print("\nStep 1: Searching internal knowledge base...")
    rag_results = rag.search(combined_query, n_results=2)
    if rag_results['success'] and rag_results['results']:
        for result in rag_results['results']:
            all_results.append({
                'source': 'Knowledge Base',
                'content': result['content'][:200],
                'relevance': result['relevance']
            })
        print(f"Found {len(rag_results['results'])} results in knowledge base")
    
    # Search web if needed
    if tavily and len(all_results) < 3:
        print("\nStep 2: Searching web for additional information...")
        web_results = tavily.search(combined_query, max_results=2)
        if web_results['success']:
            for result in web_results['results']:
                all_results.append({
                    'source': 'Web',
                    'content': result['snippet'],
                    'relevance': result.get('score', 0)
                })
            print(f"Found {len(web_results['results'])} web results")
    
    # Analyze combined results
    print(f"\nStep 3: Analyzing {len(all_results)} combined results...")
    
    # Create comprehensive report
    report = f"# Research Report: {combined_query}\n\n"
    report += f"## Summary\n"
    report += f"Total sources analyzed: {len(all_results)}\n"
    report += f"- Knowledge Base: {sum(1 for r in all_results if r['source'] == 'Knowledge Base')}\n"
    report += f"- Web Sources: {sum(1 for r in all_results if r['source'] == 'Web')}\n\n"
    
    report += f"## Findings\n\n"
    for i, result in enumerate(all_results, 1):
        report += f"### {i}. From {result['source']}\n"
        report += f"{result['content']}\n"
        report += f"*Relevance: {result['relevance']}*\n\n"
    
    # Get statistics
    all_text = " ".join([r['content'] for r in all_results])
    stats = summarizer.count_words(all_text)
    
    report += f"## Statistics\n"
    report += f"- Words: {stats['word_count']}\n"
    report += f"- Sentences: {stats['sentence_count']}\n"
    report += f"- Avg sentence length: {stats['avg_sentence_length']} words\n"
    
    # Save report
    print("\nStep 4: Saving comprehensive report...")
    save_result = writer.write_markdown(report, "comprehensive_report")
    print(f"Report saved to: {save_result['file_path']}")
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nGenerated files in: ./output/demo/")
    print("\nWorkflows demonstrated:")
    print("1. Web Research → Summarize → Save")
    print("2. RAG Search → Extract Data → Save")
    print("3. Combined Search → Analyze → Report")
    print("\nThese patterns can be used in your CrewAI agents!")
    print("="*70 + "\n")


def show_crewai_integration_example():
    """Show how to use these tools with CrewAI"""
    
    print("\n" + "="*70)
    print("CrewAI Integration Example")
    print("="*70)
    
    example_code = '''
# Example: Using tools with CrewAI agents

from crewai import Agent, Task, Crew
from tools.search_tool import TavilySearchTool
from tools.rag_tool import RAGTool
from tools.custom_tools import save_to_file, summarize_text

# Initialize tools
tavily = TavilySearchTool()
rag = RAGTool(collection_name="ai_assistant_kb")

# Define tool functions for CrewAI
def web_search(query: str) -> str:
    """Search the web for current information"""
    result = tavily.search(query, max_results=3)
    if result['success']:
        return tavily.quick_search(query)
    return "Search failed"

def knowledge_search(query: str) -> str:
    """Search internal knowledge base"""
    return rag.get_context(query, n_results=3)

def save_results(content: str, filename: str) -> str:
    """Save content to file"""
    return save_to_file(content, filename, format="md")

# Create agent with tools
research_agent = Agent(
    role='Research Specialist',
    goal='Find and compile accurate information',
    backstory='Expert researcher with access to web and knowledge base',
    tools=[web_search, knowledge_search, save_results],
    verbose=True
)

# Create task
task = Task(
    description='Research AI agents and save a summary report',
    agent=research_agent,
    expected_output='Comprehensive report on AI agents'
)

# Run crew
crew = Crew(agents=[research_agent], tasks=[task])
result = crew.kickoff()

print(result)
'''
    
    print(example_code)
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        # Run the demo
        demo_tool_integration()
        
        # Show CrewAI integration
        show_crewai_integration_example()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()

