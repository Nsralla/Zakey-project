"""
CrewAI Agent Setup
Defines and configures AI agents with their roles, goals, and tools.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Import CrewAI
from crewai import Agent, Task, Crew, Process

# Import LangChain components
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Import tools
from tools.search_tool import TavilySearchTool
from tools.rag_tool import RAGTool
from tools.custom_tools import FileWriterTool, SummaryTool, DataParserTool


class AgentTeam:
    """
    Manages the AI agent team and their configurations.
    """
    
    def __init__(self, config_path: str = "agents/agent_config.yaml"):
        """
        Initialize the agent team.
        
        Args:
            config_path: Path to agent configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize LLM with reduced token limit
        self._init_llm()
        
        # Initialize tools
        self._init_tools()
        
        # Initialize agents
        self.agents = self._create_agents()
        
        print(f"✅ Agent team initialized with {len(self.agents)} agents")
    
    def _load_config(self) -> Dict:
        """Load agent configuration from YAML file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _init_llm(self):
        """Initialize LLM with reduced token limits to stay within credits"""
        
        # Get API configuration
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL", "google/gemini-2.0-flash-exp:free")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY required in .env")
        
        # Configure LLM with reduced max_tokens to fit credit limits
        # 478 credits remaining, so use max 400 tokens per response
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            max_tokens=400,  # Reduced from default 4096 to fit within 478 credits
            temperature=0.7,
            timeout=60
        )
        
        print(f"✓ LLM configured: {model_name} (max_tokens=400)")
    
    def _init_tools(self):
        """Initialize all tools that agents will use"""
        
        # Search tools
        self.tavily = TavilySearchTool() if os.getenv("TAVILY_API_KEY") else None
        self.rag = RAGTool(collection_name="ai_assistant_kb")
        
        # Processing tools
        self.writer = FileWriterTool(output_dir="./output/agent_results")
        self.summarizer = SummaryTool(use_llm=True)  # Use extractive by default
        self.parser = DataParserTool()
        
        print(" Tools initialized")
    
    def _create_tool_functions(self) -> Dict[str, callable]:
        """
        Create tool functions that agents can use.
        These wrap our tools in LangChain tool decorators.
        """
        
        # Reference to self for closures
        tavily = self.tavily
        rag = self.rag
        writer = self.writer
        summarizer = self.summarizer
        parser = self.parser
        
        @tool
        def web_search(query: str) -> str:
            """Search the web for current information"""
            if not tavily:
                return "Web search not available (TAVILY_API_KEY not set)"
            
            try:
                result = tavily.search(query, max_results=3, include_answer=True)
                if result['success']:
                    output = f"Web Search Results for: {query}\n\n"
                    
                    if 'answer' in result:
                        output += f"Quick Answer:\n{result['answer']}\n\n"
                    
                    output += "Top Sources:\n"
                    for i, item in enumerate(result['results'], 1):
                        output += f"\n{i}. {item['title']}\n"
                        output += f"   {item['snippet']}\n"
                        output += f"   URL: {item['url']}\n"
                    
                    return output
                else:
                    return f"Search failed: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error during search: {str(e)}"
        
        @tool
        def knowledge_base_search(query: str) -> str:
            """Search internal knowledge base and get AI-synthesized answer"""
            try:
                # Use LLM-enhanced search
                result = rag.search_with_llm(query, n_results=5)
                
                if result['success']:
                    output = f"Knowledge Base Search: {query}\n\n"
                    output += "=== AI-Synthesized Answer ===\n"
                    output += result['answer'] + "\n\n"
                    
                    # Add source references
                    if result.get('sources'):
                        output += "=== Sources Used ===\n"
                        for i, source in enumerate(result['sources'], 1):
                            output += f"{i}. {source['source']} (Relevance: {source['relevance']:.1%})\n"
                    
                    # Add note if any
                    if result.get('note'):
                        output += f"\nNote: {result['note']}\n"
                    
                    return output
                else:
                    return f"No results found in knowledge base for: {query}\nError: {result.get('error', 'Unknown')}"
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        @tool
        def create_summary(text: str) -> str:
            """Create a summary of the provided text. Input should be the text to summarize."""
            try:
                max_sentences = 5
                result = summarizer.create_summary(text, max_sentences=max_sentences)
                
                if result['success']:
                    return f"Summary:\n{result['summary']}\n\n" \
                           f"(Compressed from {result['original_length']} to {result['summary_length']} characters)"
                else:
                    return f"Error creating summary: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def extract_key_points(text: str) -> str:
            """Extract key points from text. Input should be the text to analyze."""
            try:
                num_points = 5
                result = summarizer.extract_key_points(text, num_points=num_points)
                
                if result['success']:
                    output = "Key Points:\n"
                    for i, point in enumerate(result['key_points'], 1):
                        output += f"{i}. {point}\n"
                    return output
                else:
                    return f"Error extracting key points: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def parse_data(text: str) -> str:
            """Extract URLs, emails, and numbers from text. Input should be the text to parse."""
            try:
                urls = parser.extract_urls(text)
                emails = parser.extract_emails(text)
                numbers = parser.extract_numbers(text)
                
                output = "Extracted Data:\n\n"
                
                if urls:
                    output += f"URLs found ({len(urls)}):\n" + "\n".join(f"- {url}" for url in urls[:5]) + "\n\n"
                
                if emails:
                    output += f"Emails found ({len(emails)}):\n" + "\n".join(f"- {email}" for email in emails[:5]) + "\n\n"
                
                if numbers:
                    output += f"Numbers found ({len(numbers)}):\n" + "\n".join(f"- {num}" for num in numbers[:10]) + "\n\n"
                
                return output if (urls or emails or numbers) else "No structured data found"
                
            except Exception as e:
                return f"Error parsing data: {str(e)}"
        
        @tool
        def save_markdown_report(content: str) -> str:
            """Save content as a markdown report and return the content. Input should be the markdown content to save."""
            try:
                # Generate filename from first line or use default
                import re
                from datetime import datetime
                
                first_line = content.split('\n')[0]
                filename_base = re.sub(r'[^a-zA-Z0-9\s]', '', first_line)[:30]
                filename_base = filename_base.strip().replace(' ', '_').lower()
                
                if not filename_base:
                    filename_base = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                result = writer.write_markdown(content, filename_base, title=None)
                
                if result['success']:
                    # Return the full content with save confirmation at top
                    return f"[Report saved to: {result['file_path']}]\n\n{content}"
                else:
                    # Even if save failed, return the content
                    return f"[Warning: Save failed - {result.get('error', 'Unknown')}]\n\n{content}"
            except Exception as e:
                # Return content even on error
                return f"[Error saving: {str(e)}]\n\n{content}"
        
        @tool
        def save_json_data(content: str) -> str:
            """Save data as JSON file. Input should be a string representation of JSON data."""
            try:
                import json
                from datetime import datetime
                
                # Try to parse as JSON
                try:
                    data = json.loads(content)
                except:
                    # If not valid JSON, create a simple structure
                    data = {"content": content, "timestamp": datetime.now().isoformat()}
                
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                result = writer.write_json(data, filename)
                
                if result['success']:
                    return f"Data saved successfully to: {result['file_path']}"
                else:
                    return f"Error saving data: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Return dictionary of all tool functions
        return {
            'web_search': web_search,
            'knowledge_base_search': knowledge_base_search,
            'create_summary': create_summary,
            'extract_key_points': extract_key_points,
            'parse_data': parse_data,
            'save_markdown_report': save_markdown_report,
            'save_json_data': save_json_data
        }
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create Agent instances from configuration"""
        
        tool_functions = self._create_tool_functions()
        agents = {}
        
        agent_configs = self.config.get('agents', {})
        
        for agent_id, agent_config in agent_configs.items():
            # Get tools for this agent
            tool_names = agent_config.get('tools', [])
            agent_tools = [tool_functions[name] for name in tool_names if name in tool_functions]
            
            # Create agent with custom LLM
            agent = Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                tools=agent_tools,
                verbose=agent_config.get('verbose', True),
                allow_delegation=agent_config.get('allow_delegation', False),
                max_iter=agent_config.get('max_iter', 5),
                llm=self.llm  # Use our configured LLM with reduced tokens
            )
            
            agents[agent_id] = agent
            print(f"  ✓ Created {agent_id}: {agent_config['role']}")
        
        return agents
    
    def create_crew(
        self,
        tasks: List[Task],
        process: str = "sequential",
        verbose: int = 2
    ) -> Crew:
        """
        Create a Crew with the specified tasks.
        
        Args:
            tasks: List of Task objects
            process: "sequential" or "hierarchical"
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Configured Crew instance
        """
        
        # Get agents used in tasks
        task_agents = [task.agent for task in tasks]
        
        # Determine process
        process_type = Process.sequential if process == "sequential" else Process.hierarchical
        
        crew = Crew(
            agents=task_agents,
            tasks=tasks,
            process=process_type,
            verbose=verbose
        )
        
        return crew
    
    def create_task(
        self,
        agent_id: str,
        description: str,
        expected_output: str,
        context: Optional[List[Task]] = None
    ) -> Task:
        """
        Create a Task for a specific agent.
        
        Args:
            agent_id: ID of the agent (from config)
            description: Task description
            expected_output: What the task should produce
            context: Optional list of previous tasks for context
            
        Returns:
            Task instance
        """
        
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        
        task = Task(
            description=description,
            agent=self.agents[agent_id],
            expected_output=expected_output,
            context=context or []
        )
        
        return task


# Convenience function to create a basic research crew
def create_research_crew(topic: str) -> Crew:
    """
    Create a basic research crew for a given topic.
    
    Args:
        topic: The topic to research
        
    Returns:
        Configured Crew ready to run
    """
    
    team = AgentTeam()
    
    # Create tasks
    research_task = team.create_task(
        agent_id='research_agent',
        description=f"Research the topic: {topic}. Use both web search and knowledge base to gather comprehensive information. Focus on recent developments, key concepts, and practical applications.",
        expected_output="Detailed research findings with sources and key information about the topic"
    )
    
    analysis_task = team.create_task(
        agent_id='analyst_agent',
        description=f"Analyze the research findings about {topic}. Extract key points, identify important patterns, and synthesize the information into clear insights.",
        expected_output="List of key insights, patterns, and important takeaways",
        context=[research_task]
    )
    
    writing_task = team.create_task(
        agent_id='writer_agent',
        description=f"Create a comprehensive, well-structured markdown report about {topic} based on the research and analysis. Include an introduction, key findings, detailed sections, and a conclusion. After creating the report, save it AND return the full report content as your final answer.",
        expected_output="Complete markdown report with introduction, key findings, detailed sections, and conclusion",
        context=[research_task, analysis_task]
    )
    
    # Create and return crew
    crew = team.create_crew(
        tasks=[research_task, analysis_task, writing_task],
        process="sequential",
        verbose=2
    )
    
    return crew
