"""
Unit tests for Tavily Search Tool
Run with: pytest test_search_tool.py -v
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_tool import TavilySearchTool, search_web


class TestTavilySearchTool:
    """Test suite for TavilySearchTool class"""
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_api_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_initialization_with_env_key(self, mock_client):
        """Test initialization with API key from environment"""
        tool = TavilySearchTool()
        assert tool.api_key == 'test_api_key'
        mock_client.assert_called_once_with(api_key='test_api_key')
    
    @patch('tools.search_tool.TavilyClient')
    def test_initialization_with_direct_key(self, mock_client):
        """Test initialization with API key passed directly"""
        tool = TavilySearchTool(api_key='direct_key')
        assert tool.api_key == 'direct_key'
        mock_client.assert_called_once_with(api_key='direct_key')
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_without_key_raises_error(self):
        """Test that initialization without API key raises ValueError"""
        with pytest.raises(ValueError, match="Tavily API key not found"):
            TavilySearchTool()
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_search_with_valid_query(self, mock_client):
        """Test search with valid query returns formatted results"""
        # Mock the Tavily client response
        mock_response = {
            'query': 'test query',
            'answer': 'Test answer',
            'results': [
                {
                    'title': 'Test Result',
                    'url': 'https://example.com',
                    'content': 'Test content',
                    'score': 0.95
                }
            ]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        tool = TavilySearchTool()
        result = tool.search('test query')
        
        assert result['success'] is True
        assert result['query'] == 'test query'
        assert result['answer'] == 'Test answer'
        assert len(result['results']) == 1
        assert result['results'][0]['title'] == 'Test Result'
        assert 'response_time' in result
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_search_with_empty_query(self, mock_client):
        """Test search with empty query returns error"""
        tool = TavilySearchTool()
        result = tool.search('')
        
        assert result['success'] is False
        assert 'error' in result
        assert result['results'] == []
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_search_with_exception(self, mock_client):
        """Test search handles exceptions gracefully"""
        mock_client_instance = Mock()
        mock_client_instance.search.side_effect = Exception("API Error")
        mock_client.return_value = mock_client_instance
        
        tool = TavilySearchTool()
        result = tool.search('test query')
        
        assert result['success'] is False
        assert 'error' in result
        assert 'API Error' in result['error']
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_search_news(self, mock_client):
        """Test news search functionality"""
        mock_response = {
            'query': 'test news',
            'results': [
                {
                    'title': 'News Article',
                    'url': 'https://news.example.com',
                    'content': 'News content',
                    'score': 0.9,
                    'published_date': '2024-01-01'
                }
            ]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        tool = TavilySearchTool()
        result = tool.search_news('test news', days=7)
        
        assert result['success'] is True
        assert len(result['results']) > 0
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_quick_search(self, mock_client):
        """Test quick search returns string output"""
        mock_response = {
            'query': 'quick test',
            'answer': 'Quick answer',
            'results': [
                {
                    'title': 'Result 1',
                    'url': 'https://example1.com',
                    'content': 'Content 1',
                    'score': 0.9
                }
            ]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        tool = TavilySearchTool()
        result = tool.quick_search('quick test')
        
        assert isinstance(result, str)
        assert 'quick test' in result
        assert 'Result 1' in result
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_get_sources(self, mock_client):
        """Test get_sources returns list of URLs"""
        mock_response = {
            'query': 'test',
            'results': [
                {'title': 'R1', 'url': 'https://ex1.com', 'content': 'C1', 'score': 0.9},
                {'title': 'R2', 'url': 'https://ex2.com', 'content': 'C2', 'score': 0.8},
            ]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        tool = TavilySearchTool()
        sources = tool.get_sources('test', num_sources=2)
        
        assert len(sources) == 2
        assert 'https://ex1.com' in sources
        assert 'https://ex2.com' in sources
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_format_response(self, mock_client):
        """Test response formatting"""
        tool = TavilySearchTool()
        
        raw_response = {
            'query': 'test',
            'answer': 'answer',
            'results': [
                {
                    'title': 'Title',
                    'url': 'https://example.com',
                    'content': 'Content',
                    'score': 0.95,
                    'published_date': '2024-01-01'
                }
            ],
            'images': ['image1.jpg', 'image2.jpg']
        }
        
        formatted = tool._format_response(raw_response, 1.5)
        
        assert formatted['success'] is True
        assert formatted['query'] == 'test'
        assert formatted['answer'] == 'answer'
        assert formatted['response_time'] == 1.5
        assert len(formatted['results']) == 1
        assert formatted['results'][0]['title'] == 'Title'
        assert 'images' in formatted
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_search_with_custom_parameters(self, mock_client):
        """Test search with custom parameters"""
        mock_response = {'query': 'test', 'results': []}
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        tool = TavilySearchTool()
        result = tool.search(
            query='test',
            search_depth='advanced',
            max_results=10,
            include_domains=['example.com'],
            exclude_domains=['spam.com'],
            include_answer=True,
            include_images=True
        )
        
        # Verify the search was called with correct parameters
        mock_client_instance.search.assert_called_once()
        call_kwargs = mock_client_instance.search.call_args[1]
        assert call_kwargs['search_depth'] == 'advanced'
        assert call_kwargs['max_results'] == 10
        assert call_kwargs['include_domains'] == ['example.com']


class TestConvenienceFunctions:
    """Test suite for convenience functions"""
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('tools.search_tool.TavilyClient')
    def test_search_web_function(self, mock_client):
        """Test search_web convenience function"""
        mock_response = {
            'query': 'test',
            'results': [{'title': 'R', 'url': 'U', 'content': 'C', 'score': 0.9}]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        result = search_web('test', max_results=3)
        
        assert result['success'] is True
        assert result['query'] == 'test'


# Integration test (requires real API key)
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('TAVILY_API_KEY'), reason="Requires TAVILY_API_KEY")
def test_real_search():
    """Integration test with real API (requires valid API key)"""
    tool = TavilySearchTool()
    result = tool.search("Python programming", max_results=2)
    
    assert result['success'] is True
    assert len(result['results']) > 0
    assert 'url' in result['results'][0]
    assert 'title' in result['results'][0]


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])

