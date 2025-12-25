"""
MCP Server Implementation using FastAPI
Exposes AI agents and tools as API endpoints for external applications.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import project tools
from tools.search_tool import TavilySearchTool
from tools.rag_tool import RAGTool
from tools.custom_tools import FileWriterTool, SummaryTool
from agents.crew_setup import AgentTeam, create_research_crew

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent MCP Server",
    description="Model Context Protocol server exposing AI agents and tools",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tools globally
tavily = TavilySearchTool() if os.getenv("TAVILY_API_KEY") else None
rag = RAGTool(collection_name="ai_assistant_kb")
writer = FileWriterTool(output_dir="./output/mcp_results")
summarizer = SummaryTool(use_llm=False)

# Store for running tasks
task_results = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum number of results")

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[Dict[str, Any]]
    answer: Optional[str] = None

class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="Search query for knowledge base")
    n_results: int = Field(3, description="Number of results to return")

class RAGSearchResponse(BaseModel):
    success: bool
    query: str
    answer: Optional[str] = None  # LLM-synthesized answer
    results: List[Dict[str, Any]]
    total_results: int
    sources: Optional[List[Dict[str, Any]]] = None  # Source information
    note: Optional[str] = None  # Any notes (e.g., fallback to raw results)

class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Topic to research")
    save_report: bool = Field(True, description="Whether to save the report")

class ResearchResponse(BaseModel):
    success: bool
    topic: str
    result: str
    report_path: Optional[str] = None
    task_id: Optional[str] = None

class SummaryRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_sentences: int = Field(5, description="Maximum sentences in summary")

class SummaryResponse(BaseModel):
    success: bool
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

class AddDocumentRequest(BaseModel):
    content: str = Field(..., description="Document content")
    source: str = Field("api", description="Document source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AddDocumentResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "name": "AI Agent MCP Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "web_search": "/tools/web-search",
            "kb_search": "/tools/kb-search",
            "research": "/agents/research",
            "summarize": "/tools/summarize",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tools": {
            "web_search": tavily is not None,
            "knowledge_base": True,
            "agents": True,
            "summarizer": True
        }
    }


# ============================================================================
# Tool Endpoints
# ============================================================================

@app.post("/tools/web-search", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """
    Search the web using Tavily API.
    
    Returns current information from the internet.
    """
    if not tavily:
        raise HTTPException(
            status_code=503,
            detail="Web search not available. TAVILY_API_KEY not configured."
        )
    
    try:
        result = tavily.search(
            query=request.query,
            max_results=request.max_results,
            include_answer=True
        )
        
        if result['success']:
            return SearchResponse(
                success=True,
                query=result['query'],
                results=result['results'],
                answer=result.get('answer')
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/kb-search", response_model=RAGSearchResponse)
async def knowledge_base_search(request: RAGSearchRequest):
    """
    Search the internal knowledge base using RAG with LLM synthesis.
    
    Returns AI-synthesized answers based on relevant documents from the knowledge base.
    """
    try:
        # Use LLM-enhanced search
        result = rag.search_with_llm(
            query=request.query,
            n_results=request.n_results
        )
        
        if result['success']:
            # For backward compatibility, also include raw results
            raw_results = rag.search(request.query, request.n_results)
            
            return RAGSearchResponse(
                success=True,
                query=result['query'],
                answer=result.get('answer'),
                results=raw_results.get('results', []),
                total_results=len(result.get('sources', [])),
                sources=result.get('sources'),
                note=result.get('note')
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/summarize", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    """
    Create a summary of the provided text.
    
    Uses extractive summarization (no API costs).
    """
    try:
        result = summarizer.create_summary(
            text=request.text,
            max_sentences=request.max_sentences
        )
        
        if result['success']:
            return SummaryResponse(
                success=True,
                summary=result['summary'],
                original_length=result['original_length'],
                summary_length=result['summary_length'],
                compression_ratio=result['compression_ratio']
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Agent Endpoints
# ============================================================================

@app.post("/agents/research", response_model=ResearchResponse)
async def research_topic(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Run AI agents to research a topic.
    
    This uses a crew of 3 agents:
    - Research Agent: Gathers information
    - Analyst Agent: Analyzes findings
    - Writer Agent: Creates report
    
    Returns the research result and optionally saves to file.
    """
    try:
        import re
        
        # Generate task ID
        task_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create and run research crew
        crew = create_research_crew(request.topic)
        result = crew.kickoff()
        
        # Convert result to string
        result_text = str(result)
        
        # Save report if requested
        report_path = None
        if request.save_report:
            # Create filename from topic
            filename = re.sub(r'[^a-zA-Z0-9\s]', '', request.topic)[:30]
            filename = filename.strip().replace(' ', '_').lower()
            filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save as markdown
            save_result = writer.write_markdown(
                content=result_text,
                filename=filename,
                title=f"Research Report: {request.topic}"
            )
            
            if save_result['success']:
                report_path = save_result['file_path']
        
        # Store result
        task_results[task_id] = {
            'topic': request.topic,
            'result': result_text,
            'report_path': report_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return ResearchResponse(
            success=True,
            topic=request.topic,
            result=result_text,
            report_path=report_path,
            task_id=task_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Knowledge Base Management
# ============================================================================

@app.get("/kb/stats")
async def knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = rag.get_collection_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/add-document", response_model=AddDocumentResponse)
async def add_document_to_kb(request: AddDocumentRequest):
    """Add a document to the knowledge base"""
    try:
        # Add metadata
        full_metadata = {
            "source": request.source,
            "timestamp": datetime.now().isoformat(),
            **request.metadata
        }
        
        # Generate document ID
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add to RAG
        result = rag.add_documents(
            documents=[request.content],
            metadatas=[full_metadata],
            ids=[doc_id]
        )
        
        return AddDocumentResponse(
            success=True,
            message=f"Added {result} document(s) to knowledge base",
            document_id=doc_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MCP Protocol Information
# ============================================================================

@app.get("/mcp/tools")
async def list_mcp_tools():
    """
    List all available MCP tools in standard format.
    This endpoint helps clients discover available tools.
    """
    return {
        "tools": [
            {
                "name": "web_search",
                "description": "Search the web for current information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "kb_search",
                "description": "Search internal knowledge base",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "n_results": {"type": "integer", "default": 3}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "research_topic",
                "description": "Run AI agents to research a topic comprehensively",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to research"},
                        "save_report": {"type": "boolean", "default": True}
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "summarize",
                "description": "Create a summary of text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize"},
                        "max_sentences": {"type": "integer", "default": 5}
                    },
                    "required": ["text"]
                }
            }
        ]
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print(" Starting AI Agent MCP Server")
    print("="*70)
    print(f"\nServer will be available at: http://localhost:8000")
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    uvicorn.run(
        "mcp_server.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

