"""
Example: Running CrewAI Agents
Demonstrates agents working together on a research task.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.crew_setup import AgentTeam, create_research_crew


def example_1_simple_research():
    """
    Example 1: Simple research workflow
    Uses the convenience function to create a research crew
    """
    
    print("\n" + "="*70)
    print("Example 1: Simple Research Workflow")
    print("="*70 + "\n")
    
    # Define topic
    topic = "CrewAI framework for AI agents"
    
    print(f"📋 Topic: {topic}\n")
    print("Creating crew with 3 agents:")
    print("  1. Research Agent - Gathers information")
    print("  2. Analyst Agent - Extracts insights")
    print("  3. Writer Agent - Creates report\n")
    
    # Create crew
    crew = create_research_crew(topic)
    
    print("🚀 Starting crew execution...\n")
    print("-" * 70 + "\n")
    
    # Run the crew
    result = crew.kickoff()
    
    print("\n" + "-" * 70)
    print("\n✅ Crew execution complete!")
    print(f"\n📄 Final Result:\n{result}\n")
    
    return result


def example_2_custom_workflow():
    """
    Example 2: Custom workflow with specific tasks
    Shows how to create custom tasks for agents
    """
    
    print("\n" + "="*70)
    print("Example 2: Custom Workflow")
    print("="*70 + "\n")
    
    # Initialize agent team
    team = AgentTeam()
    
    # Define custom topic
    topic = "AI agent tools and frameworks comparison"
    
    print(f"📋 Topic: {topic}\n")
    
    # Create custom tasks
    print("Creating custom tasks...\n")
    
    # Task 1: Research from knowledge base only
    research_task = team.create_task(
        agent_id='research_agent',
        description=f"""
        Search the internal knowledge base for information about: {topic}
        Focus on finding detailed information about different frameworks and tools.
        Use the knowledge_base_search tool to gather information.
        """,
        expected_output="Comprehensive findings from knowledge base with sources"
    )
    
    # Task 2: Analyze and compare
    analysis_task = team.create_task(
        agent_id='analyst_agent',
        description=f"""
        Analyze the research findings about {topic}.
        Extract key points and create a comparison.
        Identify strengths and use cases for each tool/framework.
        Use the extract_key_points tool to structure your analysis.
        """,
        expected_output="Structured analysis with key points and comparisons",
        context=[research_task]
    )
    
    # Task 3: Create summary report
    writing_task = team.create_task(
        agent_id='writer_agent',
        description=f"""
        Based on the research and analysis, create a concise summary report about {topic}.
        Structure the report with:
        - Introduction
        - Key Findings
        - Comparison Table
        - Conclusion
        
        Use the save_markdown_report tool to save the final report.
        """,
        expected_output="Markdown report saved successfully",
        context=[research_task, analysis_task]
    )
    
    # Create crew with custom tasks
    crew = team.create_crew(
        tasks=[research_task, analysis_task, writing_task],
        process="sequential",
        verbose=2
    )
    
    print("🚀 Starting custom workflow...\n")
    print("-" * 70 + "\n")
    
    # Execute
    result = crew.kickoff()
    
    print("\n" + "-" * 70)
    print("\n✅ Custom workflow complete!")
    print(f"\n📄 Result:\n{result}\n")
    
    return result


def example_3_single_agent():
    """
    Example 3: Single agent task
    Shows how to use just one agent for a simple task
    """
    
    print("\n" + "="*70)
    print("Example 3: Single Agent Task")
    print("="*70 + "\n")
    
    # Initialize agent team
    team = AgentTeam()
    
    # Create a simple research task
    task = team.create_task(
        agent_id='research_agent',
        description="""
        Search for information about 'RAG (Retrieval Augmented Generation)' 
        using the knowledge base. Provide a clear explanation with examples.
        """,
        expected_output="Clear explanation of RAG with examples"
    )
    
    # Create crew with single agent
    crew = team.create_crew(
        tasks=[task],
        process="sequential",
        verbose=2
    )
    
    print("🚀 Running single agent task...\n")
    print("-" * 70 + "\n")
    
    # Execute
    result = crew.kickoff()
    
    print("\n" + "-" * 70)
    print("\n✅ Single agent task complete!")
    print(f"\n📄 Result:\n{result}\n")
    
    return result


def example_4_web_and_kb_combined():
    """
    Example 4: Combining web search and knowledge base
    Shows how to use multiple tools together
    """
    
    print("\n" + "="*70)
    print("Example 4: Web Search + Knowledge Base")
    print("="*70 + "\n")
    
    # Initialize agent team
    team = AgentTeam()
    
    topic = "Latest developments in AI agents"
    
    # Create task that uses both web and KB
    research_task = team.create_task(
        agent_id='research_agent',
        description=f"""
        Research '{topic}' using BOTH web search and knowledge base:
        
        1. First, search the knowledge base for foundational information
        2. Then, use web search to find latest news and developments
        3. Combine both sources to provide comprehensive findings
        
        Make sure to cite sources from both searches.
        """,
        expected_output="Comprehensive research combining knowledge base and web sources"
    )
    
    # Create analysis task
    analysis_task = team.create_task(
        agent_id='analyst_agent',
        description=f"""
        Analyze the combined research findings about {topic}.
        
        Create a structured summary that includes:
        - Core concepts (from knowledge base)
        - Recent developments (from web search)
        - Key insights and trends
        
        Extract key points to make the information easy to digest.
        """,
        expected_output="Structured analysis with key insights",
        context=[research_task]
    )
    
    # Create crew
    crew = team.create_crew(
        tasks=[research_task, analysis_task],
        process="sequential",
        verbose=2
    )
    
    print(f"📋 Topic: {topic}\n")
    print("🚀 Starting combined search workflow...\n")
    print("-" * 70 + "\n")
    
    # Execute
    result = crew.kickoff()
    
    print("\n" + "-" * 70)
    print("\n✅ Combined search complete!")
    print(f"\n📄 Result:\n{result}\n")
    
    return result


def interactive_mode():
    """
    Interactive mode: Let user specify topic
    """
    
    print("\n" + "="*70)
    print("Interactive Mode - Research Any Topic")
    print("="*70 + "\n")
    
    topic = input("Enter a topic to research: ").strip()
    
    if not topic:
        print("No topic provided. Using default topic.")
        topic = "AI agents and autonomous systems"
    
    print(f"\n📋 Researching: {topic}\n")
    
    # Use the convenience function
    crew = create_research_crew(topic)
    
    print("🚀 Starting research...\n")
    print("-" * 70 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "-" * 70)
    print("\n✅ Research complete!")
    print(f"\n📄 Result:\n{result}\n")
    
    return result


def main():
    """Main function to run examples"""
    
    print("\n" + "="*70)
    print("CrewAI Agents - Working Examples")
    print("="*70)
    
    # Check for required API keys
    warnings = []
    if not os.getenv("TAVILY_API_KEY"):
        warnings.append("⚠️  TAVILY_API_KEY not set - web search will not work")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        warnings.append("⚠️  No LLM API key set - agents may not function properly")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    # Show menu
    print("\nAvailable Examples:")
    print("1. Simple Research Workflow (3 agents)")
    print("2. Custom Workflow")
    print("3. Single Agent Task")
    print("4. Web Search + Knowledge Base Combined")
    print("5. Interactive Mode (choose your own topic)")
    print("0. Exit")
    
    choice = input("\nSelect example (1-5) or 0 to exit: ").strip()
    
    if choice == "1":
        example_1_simple_research()
    elif choice == "2":
        example_2_custom_workflow()
    elif choice == "3":
        example_3_single_agent()
    elif choice == "4":
        example_4_web_and_kb_combined()
    elif choice == "5":
        interactive_mode()
    elif choice == "0":
        print("\nExiting...")
    else:
        print("\nInvalid choice. Running default example...")
        example_1_simple_research()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

