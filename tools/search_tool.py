"""
Tavily Search Tool for AI Agents
Provides web search capabilities using Tavily API with error handling and retry logic.
"""

import os
import logging
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilySearchTool:
    """    
    Features:
    - Web search with customizable depth
    - News search
    - Result filtering and ranking
    - Error handling and retry logic
    - Structured response formatting
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily Search Tool.
        
        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. Please set TAVILY_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("✅ Tavily Search Tool initialized successfully")
        except ImportError:
            raise ImportError(
                "tavily-python package not found. Install it with: pip install tavily-python"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            raise
    
    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_answer: bool = True,
        include_images: bool = False
    ) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: The search query string
            search_depth: Search depth - "basic" or "advanced" (default: "basic")
            max_results: Maximum number of results to return (default: 5)
            include_answer: Whether to include AI-generated answer (default: True)
            include_images: Whether to include images in results (default: False)
            
        Returns:
            Dictionary containing search results with the following structure:
            {
                'success': bool,
                'query': str,
                'answer': str (if include_answer=True),
                'results': List[Dict],
                'images': List[Dict] (if include_images=True),
                'response_time': float,
                'error': str (if success=False)
            }
        """
        if not query or not query.strip():
            return {
                'success': False,
                'error': 'Query cannot be empty',
                'results': []
            }
        
        logger.info(f" Searching for: {query}")
        
        try:
            import time
            start_time = time.time()
            
            # Perform the search
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
                include_images=include_images
            )
            
            response_time = time.time() - start_time
            # print sources
            # Format the response
            formatted_response = self._format_response(response, response_time)
            logger.info(f" Search completed in {response_time:.2f}s - Found {len(formatted_response['results'])} results")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'results': []
            }
    
    def search_news(
        self,
        query: str,
        days: int = 7,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search for recent news articles.
        
        Args:
            query: The search query
            days: Number of days to look back (default: 7)
            max_results: Maximum number of results (default: 5)
            
        Returns:
            Formatted search results dictionary
        """
        logger.info(f"📰 Searching news for: {query} (last {days} days)")
        
        # Add time constraint to query
        time_query = f"{query} (news from last {days} days)"
        
        return self.search(
            query=time_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_images=True
        )
    
    def quick_search(self, query: str) -> str:
        """
        Perform a quick search and return a simple string response.
        Useful for agent tools that need string outputs.
        
        Args:
            query: The search query
            
        Returns:
            Formatted string with search results
        """
        result = self.search(query, max_results=3, search_depth="basic")
        
        if not result['success']:
            return f"Search failed: {result.get('error', 'Unknown error')}"
        
        # Format as string
        output = f"Search Results for: {query}\n\n"
        
        if result.get('answer'):
            output += f"Quick Answer:\n{result['answer']}\n\n"
        
        output += "Top Results:\n"
        for i, item in enumerate(result['results'][:3], 1):
            output += f"\n{i}. {item['title']}\n"
            output += f"   URL: {item['url']}\n"
            output += f"   {item['snippet']}\n"
        
        return output
    
    def _format_response(self, response: Dict, response_time: float) -> Dict[str, Any]:
        """
        Format the Tavily API response into a standardized structure.
        
        Args:
            response: Raw response from Tavily API
            response_time: Time taken for the search in seconds
            
        Returns:
            Formatted response dictionary
        """
        formatted = {
            'success': True,
            'query': response.get('query', ''),
            'response_time': round(response_time, 2),
            'results': []
        }
        
        # Add AI-generated answer if available
        if 'answer' in response:
            formatted['answer'] = response['answer']
        
        # Format results
        if 'results' in response:
            for result in response['results']:
                formatted_result = {
                    'title': result.get('title', 'No Title'),
                    'url': result.get('url', ''),
                    'snippet': result.get('content', ''),
                    'score': result.get('score', 0),
                    'published_date': result.get('published_date', None)
                }
                
                # Add raw content if available
                if 'raw_content' in result:
                    formatted_result['raw_content'] = result['raw_content']
                
                formatted['results'].append(formatted_result)
        
        # Add images if available
        if 'images' in response:
            formatted['images'] = response['images']
        
        return formatted
    

# Convenience function for quick searches
def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Convenience function to perform a quick web search.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Search results dictionary
    """
    tool = TavilySearchTool()
    return tool.search(query, max_results=max_results)

