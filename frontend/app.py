"""
AI Agent Research Assistant - Streamlit Frontend
A beautiful, user-friendly interface for your AI agent system.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.crew_setup import create_research_crew, AgentTeam
from tools.search_tool import TavilySearchTool
from tools.rag_tool import RAGTool
from tools.custom_tools import FileWriterTool, SummaryTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ZAKEY - AI Research Assistant",
    page_icon="⚡",  # Lightning bolt for AI/intelligence
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None


# Initialize tools (with error handling)
@st.cache_resource
def init_tools():
    """Initialize tools and cache them"""
    tools = {
        'tavily': None,
        'rag': None,
        'writer': None,
        'summarizer': None
    }
    
    try:
        if os.getenv("TAVILY_API_KEY"):
            tools['tavily'] = TavilySearchTool()
    except Exception as e:
        st.sidebar.warning(f"Tavily not available: {e}")
    
    try:
        tools['rag'] = RAGTool(collection_name="ai_assistant_kb")
    except Exception as e:
        st.sidebar.warning(f"RAG not available: {e}")
    
    try:
        tools['writer'] = FileWriterTool(output_dir="./output/streamlit_results")
        tools['summarizer'] = SummaryTool(use_llm=True)
    except Exception as e:
        st.sidebar.warning(f"Tools error: {e}")
    
    return tools


def run_research_agents(topic: str, mode: str = "full", stream_container=None):
    """
    Run the AI agent crew on a topic with optional streaming support.
    
    Args:
        topic: The research topic
        mode: 'full' for all 3 agents, 'quick' for just research agent
        stream_container: Streamlit container to stream output to (if provided)
    
    Returns:
        Final result text
    """
    try:
        # Enable streaming if container is provided
        enable_streaming = stream_container is not None
        
        if mode == "full":
            # Full research with all 3 agents
            crew = create_research_crew(topic, enable_streaming=enable_streaming)
            
            if enable_streaming and hasattr(crew, '_streaming_handler'):
                # Run in a separate thread and stream tokens
                result_container = {'result': None, 'error': None}
                
                def run_crew():
                    try:
                        raw_result = crew.kickoff()
                        # Extract actual content immediately to avoid AgentFinish issues
                        if hasattr(raw_result, 'return_values'):
                            # AgentFinish object
                            return_vals = raw_result.return_values
                            if isinstance(return_vals, dict):
                                result_container['result'] = return_vals.get('output', str(return_vals))
                            else:
                                result_container['result'] = str(return_vals)
                        elif hasattr(raw_result, 'raw'):
                            result_container['result'] = raw_result.raw
                        elif hasattr(raw_result, 'output'):
                            result_container['result'] = raw_result.output
                        else:
                            result_container['result'] = raw_result
                    except ValueError as e:
                        # Handle AgentFinish error from CrewAI when streaming
                        if "Unexpected output type from agent" in str(e) and "AgentFinish" in str(e):
                            # Extract result from streaming handler's collected tokens
                            if crew._streaming_handler and crew._streaming_handler.tokens:
                                result_container['result'] = crew._streaming_handler.get_tokens()
                            else:
                                result_container['result'] = "Agent completed but result extraction failed."
                        else:
                            result_container['error'] = e
                    except Exception as e:
                        result_container['error'] = e
                
                # Start crew execution in background thread
                import threading
                crew_thread = threading.Thread(target=run_crew)
                crew_thread.start()
                
                # Stream tokens as they arrive
                full_text = ""
                while crew_thread.is_alive() or not crew._streaming_handler.token_queue.empty():
                    try:
                        token = crew._streaming_handler.token_queue.get(timeout=0.1)
                        if token is None:  # End of stream
                            crew._streaming_handler.token_queue.task_done()
                            continue
                        full_text += token
                        stream_container.markdown(full_text)
                        crew._streaming_handler.token_queue.task_done()
                    except:
                        pass
                
                crew_thread.join()
                
                if result_container['error']:
                    raise result_container['error']
                    
                result = result_container['result']
            else:
                result = crew.kickoff()
                
        elif mode == "quick":
            # Quick research with just research agent
            team = AgentTeam(enable_streaming=enable_streaming)
            task = team.create_task(
                agent_id='research_agent',
                description=f"Quickly research: {topic}. Provide key information.",
                expected_output="Concise research findings"
            )
            crew = team.create_crew(tasks=[task], process="sequential", verbose=1)
            
            if enable_streaming and team.streaming_handler:
                # Run in a separate thread and stream tokens
                result_container = {'result': None, 'error': None}
                
                def run_crew():
                    try:
                        raw_result = crew.kickoff()
                        # Extract actual content immediately to avoid AgentFinish issues
                        if hasattr(raw_result, 'return_values'):
                            # AgentFinish object
                            return_vals = raw_result.return_values
                            if isinstance(return_vals, dict):
                                result_container['result'] = return_vals.get('output', str(return_vals))
                            else:
                                result_container['result'] = str(return_vals)
                        elif hasattr(raw_result, 'raw'):
                            result_container['result'] = raw_result.raw
                        elif hasattr(raw_result, 'output'):
                            result_container['result'] = raw_result.output
                        else:
                            result_container['result'] = raw_result
                    except ValueError as e:
                        # Handle AgentFinish error from CrewAI when streaming
                        if "Unexpected output type from agent" in str(e) and "AgentFinish" in str(e):
                            # Extract result from streaming handler's collected tokens
                            if team.streaming_handler and team.streaming_handler.tokens:
                                result_container['result'] = team.streaming_handler.get_tokens()
                            else:
                                result_container['result'] = "Agent completed but result extraction failed."
                        else:
                            result_container['error'] = e
                    except Exception as e:
                        result_container['error'] = e
                
                import threading
                crew_thread = threading.Thread(target=run_crew)
                crew_thread.start()
                
                # Stream tokens as they arrive
                full_text = ""
                while crew_thread.is_alive() or not team.streaming_handler.token_queue.empty():
                    try:
                        token = team.streaming_handler.token_queue.get(timeout=0.1)
                        if token is None:
                            team.streaming_handler.token_queue.task_done()
                            continue
                        full_text += token
                        stream_container.markdown(full_text)
                        team.streaming_handler.token_queue.task_done()
                    except:
                        pass
                
                crew_thread.join()
                
                if result_container['error']:
                    raise result_container['error']
                    
                result = result_container['result']
            else:
                result = crew.kickoff()
        else:
            result = "Invalid mode"
        
        # Extract actual content from CrewOutput object or AgentFinish
        # Handle AgentFinish objects (common with streaming)
        if hasattr(result, 'return_values'):
            # AgentFinish object from LangChain
            return_values = result.return_values
            if isinstance(return_values, dict):
                # Try to get output from common keys
                return str(return_values.get('output', return_values.get('result', str(return_values))))
            else:
                return str(return_values)
        elif hasattr(result, 'raw'):
            # CrewAI 0.5.0+ uses .raw attribute
            return str(result.raw)
        elif hasattr(result, 'result'):
            # Some versions use .result
            return str(result.result)
        elif hasattr(result, 'output'):
            # Or .output
            return str(result.output)
        else:
            # Fallback to string conversion
            return str(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error running agents: {str(e)}\n\nDetails:\n{error_details}"


def quick_search(query: str, search_type: str):
    """Quick search without agents"""
    tools = init_tools()
    
    if search_type == "Web Search" and tools['tavily']:
        result = tools['tavily'].search(query, max_results=3)
        if result['success']:
            output = ""
            if 'answer' in result:
                output += f"**Quick Answer:**\n{result['answer']}\n\n"
            output += "**Top Results:**\n"
            for i, item in enumerate(result['results'], 1):
                output += f"{i}. **{item['title']}**\n   {item['snippet']}\n   [Source]({item['url']})\n\n"
            return output
        return "Search failed"
    
    elif search_type == "Knowledge Base" and tools['rag']:
        # Use LLM-enhanced RAG search
        result = tools['rag'].search_with_llm(query, n_results=5)
        
        if result['success']:
            output = "**AI-Synthesized Answer:**\n\n"
            output += result['answer'] + "\n\n"
            
            # Add sources
            if result.get('sources'):
                output += "---\n**Sources Used:**\n\n"
                for i, source in enumerate(result['sources'], 1):
                    output += f"{i}. {source['source']} (Relevance: {source['relevance']:.2%})\n"
                    output += f"   _{source['content_preview']}_\n\n"
            
            # Add note if LLM synthesis failed
            if result.get('note'):
                output += f"\n_Note: {result['note']}_\n"
            
            return output
        return "No results found in knowledge base"
    
    return "Search tool not available"


def extract_final_answer(text: str) -> str:
    """
    Extract the Final Answer from agent output, removing the thinking process.
    
    Args:
        text: Raw agent output with thoughts, actions, and final answer
        
    Returns:
        Clean final answer text, or original text if no final answer found
    """
    import re
    
    # Try to find "Final Answer:" and extract everything after it
    final_answer_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1).strip()
    
    # If no "Final Answer:" found, return original
    return text


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Logo with Icon
    st.markdown(
        """
        <div style='text-align: center; padding: 1.5rem 0 2rem 0; border-bottom: 2px solid #1E88E5;'>
            <svg width="80" height="80" viewBox="0 0 80 80" style="margin-bottom: 0.5rem;">
                <circle cx="40" cy="40" r="35" fill="none" stroke="#1E88E5" stroke-width="3"/>
                <path d="M 25 28 L 55 28 L 25 52 L 55 52" 
                      stroke="#1E88E5" stroke-width="4" stroke-linecap="round" 
                      fill="none"/>
                <circle cx="25" cy="28" r="3" fill="#1E88E5"/>
                <circle cx="55" cy="28" r="3" fill="#1E88E5"/>
                <circle cx="25" cy="52" r="3" fill="#1E88E5"/>
                <circle cx="55" cy="52" r="3" fill="#1E88E5"/>
                <circle cx="40" cy="40" r="4" fill="#1E88E5" opacity="0.6"/>
                <line x1="20" y1="40" x2="28" y2="40" stroke="#1E88E5" stroke-width="2" opacity="0.4"/>
                <line x1="52" y1="40" x2="60" y2="40" stroke="#1E88E5" stroke-width="2" opacity="0.4"/>
            </svg>
            <div style='font-size: 2.5rem; font-weight: 800; color: #1E88E5; letter-spacing: 0.15em; margin-top: 0.5rem;'>
                ZAKEY
            </div>
            <div style='font-size: 0.85rem; color: #666; margin-top: 0.3rem; font-weight: 500; letter-spacing: 0.05em;'>
                AI RESEARCH ASSISTANT
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Settings")
    
    # Mode selection
    operation_mode = st.radio(
        "Operation Mode",
        ["Quick Search", "AI Agent Research"],
        help="Quick Search: Fast results from tools\nAI Agent Research: Full multi-agent analysis"
    )
    
    st.markdown("---")
    
    # Research settings (only for agent mode)
    if operation_mode == "AI Agent Research":
        st.markdown("### Agent Settings")
        research_depth = st.selectbox(
            "Research Depth",
            ["Quick", "Full"],
            help="Quick: Research agent only\nFull: All 3 agents (research, analysis, writing)"
        )
        
       
        search_type = None  # Not used in agent mode
    else:
        st.markdown("### Search Settings")
        search_type = st.selectbox(
            "Search Type",
            ["Knowledge Base", "Web Search"],
            help="Choose where to search"
        )
        research_depth = None  # Not used in search mode
        save_report = False  # Not used in search mode
    
    st.markdown("---")
    
    # Tool status
    st.markdown("### Tool Status")
    tools = init_tools()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Web Search", "Active" if tools['tavily'] else "Inactive")
        st.metric("Knowledge Base", "Active" if tools['rag'] else "Inactive")
    with col2:
        st.metric("AI Agents", "Active")
        st.metric("Summarizer", "Active" if tools['summarizer'] else "Inactive")
    
    st.markdown("---")
    
    # History
    st.markdown("### History")
    if st.session_state.history:
        st.write(f"Total queries: {len(st.session_state.history)}")
        
        if st.button("Show History"):
            st.session_state.show_history = True
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
    else:
        st.info("No history yet")
    
    st.markdown("---")
    
    # About
    with st.expander("About ZAKEY"):
        st.markdown("""
        ### ZAKEY - Intelligent Research Platform
        
        **What is ZAKEY?**  
        ZAKEY is an advanced AI research assistant that uses multiple specialized agents 
        working together to provide comprehensive, accurate research on any topic.
        
        **Key Features:**
        - **Multi-Agent System**: Research, Analysis, and Writing agents collaborate
        - **Dual Search**: Web search (Tavily) + Knowledge base (RAG)
        - **Intelligent Processing**: AI-powered analysis and summarization
        - **Professional Reports**: Well-structured, citation-backed outputs
        
        **Technology Stack:**
        - CrewAI for multi-agent orchestration
        - Tavily API for web search
        - RAG (Retrieval Augmented Generation) for knowledge base
        - LLMs for intelligent analysis
        - Streamlit for user interface
        
        **Use Cases:**
        - Academic research
        - Market analysis
        - Technical documentation
        - Competitive intelligence
        - Literature reviews
        
        **Version:** 1.0.0  
        **Developer:** Zakey Project Team
        """)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<p class="main-header">ZAKEY AI Research Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Multi-Agent System for Comprehensive Research and Analysis</p>', unsafe_allow_html=True)

# Mission Statement
st.info("""
**About ZAKEY:** An advanced AI-powered research platform that combines multiple intelligent agents 
to provide comprehensive, accurate, and well-structured research on any topic. ZAKEY leverages 
web search, knowledge base retrieval, and multi-agent collaboration to deliver professional-grade 
research reports in minutes.
""")

# Main input area
st.markdown("### Research Query")

col1, col2 = st.columns([4, 1])

with col1:
    if operation_mode == "AI Agent Research":
        user_input = st.text_input(
            "Enter your research topic:",
            placeholder="e.g., AI agents, Machine Learning trends, CrewAI framework...",
            label_visibility="collapsed"
        )
    else:
        user_input = st.text_input(
            "Enter your search query:",
            placeholder="e.g., What is CrewAI? How does RAG work?...",
            label_visibility="collapsed"
        )

with col2:
    submit_button = st.button("Start Research", use_container_width=True)

# Quick examples
st.markdown("**Quick examples:**")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("What is CrewAI?"):
        user_input = "What is CrewAI framework?"
        submit_button = True

with example_col2:
    if st.button("AI Agent Tools"):
        user_input = "AI agent tools and frameworks"
        submit_button = True

with example_col3:
    if st.button("RAG Systems"):
        user_input = "How does RAG work?"
        submit_button = True

st.markdown("---")

# Process input
if submit_button and user_input:
    
    # Add to history
    st.session_state.history.append({
        'timestamp': datetime.now().isoformat(),
        'query': user_input,
        'mode': operation_mode
    })
    
    if operation_mode == "AI Agent Research":
        # AI Agent Mode
        st.markdown("### AI Agents Working...")
        
        progress_text = "Research in progress. Please wait..."
        progress_bar = st.progress(0, text=progress_text)
        
        with st.spinner("Research Agent gathering information..."):
            progress_bar.progress(33, text="Research Agent gathering information...")
            
            # Run agents with streaming enabled
            mode = "quick" if research_depth == "Quick" else "full"
            
            # Show streaming output in real-time
            st.markdown("#### Live Agent Thinking (Streaming)")
            streaming_expander = st.expander("View live agent process (raw output)", expanded=False)
            with streaming_expander:
                streaming_placeholder = st.empty()
                result = run_research_agents(user_input, mode=mode, stream_container=streaming_placeholder)
            
            if research_depth == "Full":
                progress_bar.progress(66, text="Analyst Agent processing data...")
                progress_bar.progress(100, text="Writer Agent creating report...")
            else:
                progress_bar.progress(100, text="Research complete!")
        
        progress_bar.empty()
        
        # Extract clean final answer for main display
        clean_result = extract_final_answer(result)
        
        # Only display results if we have content
        if clean_result and clean_result.strip():
            # Display results
            st.markdown('<div class="success-box">Research Complete!</div>', unsafe_allow_html=True)
            
            st.markdown("### Research Results")
            
            # Create tabs for different views
            tab1, tab2= st.tabs(["Report", "Raw Output",])
            
            with tab1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(clean_result)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.text_area("Raw Output", result, height=300)
                if st.button("Copy to Clipboard"):
                    st.code(result)
            
          
            st.session_state.current_result = clean_result
        else:
            st.error("No results were generated. Please try again.")
    
    else:
        # Quick Search Mode
        if search_type:  # Make sure search_type exists
            st.markdown(f"### {search_type} Results")
            
            with st.spinner(f"Searching {search_type.lower()}..."):
                result = quick_search(user_input, search_type)
            
            st.markdown('<div class="info-box">Search Complete!</div>', unsafe_allow_html=True)
            
            # Display results only if we have content
            if result and result.strip():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.session_state.current_result = result
            else:
                st.warning("No results found. Please try a different query.")
        else:
            st.error("Search type not selected. Please select a search type in the sidebar.")

elif submit_button and not user_input:
    st.warning("Please enter a research topic or search query")

# Display history if requested
if st.session_state.get('show_history', False):
    st.markdown("---")
    st.markdown("###  Search History")
    
    for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
        with st.expander(f"{i}. {item['query']} ({item['mode']})"):
            st.write(f"**Time:** {item['timestamp']}")
            st.write(f"**Query:** {item['query']}")
            st.write(f"**Mode:** {item['mode']}")
    
    if st.button("Close History"):
        st.session_state.show_history = False

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.5rem; font-weight: bold; color: #1E88E5; margin-bottom: 0.5rem;'>ZAKEY</p>
    <p style='font-size: 1.1rem; margin-bottom: 1rem;'><strong>Intelligent AI Research Assistant</strong></p>
    <p style='margin-bottom: 0.5rem;'>Multi-Agent System for Comprehensive Research & Analysis</p>
    <p style='font-size: 0.9rem; color: #999;'>Powered by CrewAI, Tavily, RAG, and Advanced LLMs</p>
    <p style='font-size: 0.85rem; color: #999; margin-top: 1rem;'>Version 1.0.0 | © 2024 Zakey Project</p>
</div>
""", unsafe_allow_html=True)

