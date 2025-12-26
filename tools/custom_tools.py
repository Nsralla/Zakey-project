"""
Custom Tools for AI Agents
Provides utility functions for file operations, data processing, and content generation.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from langchain_openai import ChatOpenAI


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileWriterTool:
    """
    Tool for writing content to files in various formats.
    Useful for saving agent outputs, reports, and analysis results.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize File Writer Tool.
        
        Args:
            output_dir: Directory where files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileWriterTool initialized. Output directory: {self.output_dir}")
    
    def write_text(
        self,
        content: str,
        filename: str,
        append: bool = False
    ) -> Dict[str, Any]:
        """
        Write text content to a file.
        
        Args:
            content: Text content to write
            filename: Name of the file
            append: If True, append to existing file. If False, overwrite.
            
        Returns:
            Dictionary with status and file path
        """
        if not content:
            return {
                'success': False,
                'error': 'Content cannot be empty',
                'file_path': None
            }
        
        try:
            # Ensure filename has .txt extension if no extension provided
            if not Path(filename).suffix:
                filename = f"{filename}.txt"
            
            file_path = self.output_dir / filename
            mode = 'a' if append else 'w'
            
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
                if append:
                    f.write('\n')  # Add newline when appending
            
            logger.info(f"Written to file: {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'size_bytes': file_path.stat().st_size,
                'mode': 'appended' if append else 'created'
            }
            
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def write_json(
        self,
        data: Union[Dict, List],
        filename: str,
        pretty: bool = True
    ) -> Dict[str, Any]:
        """
        Write data to a JSON file.
        
        Args:
            data: Dictionary or list to write
            filename: Name of the file
            pretty: If True, format with indentation
            
        Returns:
            Dictionary with status and file path
        """
        try:
            # Ensure filename has .json extension
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            file_path = self.output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, ensure_ascii=False)
            
            logger.info(f"JSON written to: {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'size_bytes': file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error writing JSON: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def write_csv(
        self,
        data: List[Dict],
        filename: str,
        fieldnames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Write data to a CSV file.
        
        Args:
            data: List of dictionaries to write
            filename: Name of the file
            fieldnames: Optional list of field names for CSV header
            
        Returns:
            Dictionary with status and file path
        """
        if not data:
            return {
                'success': False,
                'error': 'Data cannot be empty',
                'file_path': None
            }
        
        try:
            # Ensure filename has .csv extension
            if not filename.endswith('.csv'):
                filename = f"{filename}.csv"
            
            file_path = self.output_dir / filename
            
            # Get fieldnames from first dict if not provided
            if fieldnames is None and isinstance(data[0], dict):
                fieldnames = list(data[0].keys())
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"CSV written to: {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'rows_written': len(data),
                'size_bytes': file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error writing CSV: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def write_markdown(
        self,
        content: str,
        filename: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Write content to a Markdown file with optional title.
        
        Args:
            content: Markdown content
            filename: Name of the file
            title: Optional title to prepend
            
        Returns:
            Dictionary with status and file path
        """
        try:
            # Ensure filename has .md extension
            if not filename.endswith('.md'):
                filename = f"{filename}.md"
            
            file_path = self.output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if title:
                    f.write(f"# {title}\n\n")
                    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write("---\n\n")
                f.write(content)
            
            logger.info(f"Markdown written to: {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'size_bytes': file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error writing Markdown: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }


class SummaryTool:
    """
    Tool for creating summaries and extracting key information from text.
    Can work with or without LLM integration.
    """
    
    def __init__(self, use_llm: bool = True, llm_client=None):
        """
        Initialize Summary Tool.
        
        Args:
            use_llm: If True, use LLM for summaries. If False, use extractive methods.
            llm_client: OpenAI client instance (required if use_llm=True)
        """
        self.use_llm = use_llm
        self.llm_client = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("MODEL", "google/gemma-3n-e2b-it:free"),
            max_tokens=500,  # Reduced to stay within credit limits
            temperature=0.3
        )
        
        
        logger.info(f"SummaryTool initialized (LLM: {self.use_llm})")
    
    def create_summary(
        self,
        text: str,
        max_sentences: int = 5,
        style: str = "concise"
    ) -> Dict[str, Any]:
        """
        Create a summary of the provided text.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum number of sentences in summary
            style: Summary style - "concise", "detailed", or "bullet_points"
            
        Returns:
            Dictionary with summary and metadata
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'Text cannot be empty',
                'summary': None
            }
        
        try:
            if self.use_llm:
                summary = self._llm_summary(text, max_sentences, style)
                logger.info(f"LLM summary: {summary}")
            else:
                summary = self._extractive_summary(text, max_sentences)
                logger.info(f"Extractive summary: {summary}")
            
            return {
                'success': True,
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'compression_ratio': round(len(summary) / len(text), 2),
                'method': 'llm' if self.use_llm else 'extractive'
            }
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': None
            }
    
    def _llm_summary(
        self,
        text: str,
        max_sentences: int,
        style: str
    ) -> str:
        """Create summary using LLM"""

        logger.info(f"Entering LLM summary")
        style_prompts = {
            "concise": "Create a brief, concise summary",
            "detailed": "Create a detailed, comprehensive summary",
            "bullet_points": "Create a summary as bullet points"
        }
        
        prompt = f"{style_prompts.get(style, 'Summarize')} of the following text in {max_sentences} sentences or less:\n\n{text}"
        
        try:
            response = self.llm_client.invoke(prompt)
            logger.info(f"LLM response: {response.content.strip()}")
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            # Fallback to extractive
            return self._extractive_summary(text, max_sentences)
    
    def _extractive_summary(self, text: str, max_sentences: int) -> str:
        """Create summary by extracting key sentences"""
        
        logger.info(f"Entering extractive summary")
        # Split into sentences
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring: prefer sentences with important words
        important_words = {'key', 'important', 'main', 'primary', 'essential', 
                          'critical', 'significant', 'major', 'crucial'}
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for word in important_words if word in sentence.lower())
            scored_sentences.append((score, sentence))
        
        # Sort by score and get top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    def extract_key_points(self, text: str, num_points: int = 5) -> Dict[str, Any]:
        """
        Extract key points from text.
        
        Args:
            text: Text to analyze
            num_points: Number of key points to extract
            
        Returns:
            Dictionary with key points list
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'Text cannot be empty',
                'key_points': []
            }
        
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
            
            # Get sentences with questions or key phrases
            key_indicators = ['what', 'how', 'why', 'when', 'where', 'important', 
                            'key', 'main', 'essential', 'critical']
            
            key_sentences = []
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in key_indicators):
                    key_sentences.append(sentence)
            
            # Limit to requested number
            key_points = key_sentences[:num_points]
            
            if not key_points:
                # If no key sentences found, take first few
                key_points = sentences[:num_points]
            
            return {
                'success': True,
                'key_points': key_points,
                'num_points': len(key_points)
            }
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return {
                'success': False,
                'error': str(e),
                'key_points': []
            }
    
    def count_words(self, text: str) -> Dict[str, Any]:
        """
        Get word count and basic statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'avg_word_length': round(sum(len(w) for w in words) / len(words), 2) if words else 0,
            'avg_sentence_length': round(len(words) / len(sentences), 2) if sentences else 0
        }


class DataParserTool:
    """
    Tool for extracting and structuring data from text.
    Useful for processing agent outputs into structured formats.
    """
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text.
        
        Args:
            text: Text containing URLs
            
        Returns:
            List of URLs found
        """
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        logger.info(f"Extracted {len(urls)} URLs")
        return urls
    
    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text.
        
        Args:
            text: Text containing emails
            
        Returns:
            List of email addresses found
        """
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        logger.info(f"Extracted {len(emails)} email addresses")
        return emails
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract numbers from text.
        
        Args:
            text: Text containing numbers
            
        Returns:
            List of numbers found
        """
        import re
        number_pattern = r'-?\d+\.?\d*'
        numbers = [float(n) for n in re.findall(number_pattern, text)]
        logger.info(f"Extracted {len(numbers)} numbers")
        return numbers
    
    def parse_key_value_pairs(self, text: str) -> Dict[str, str]:
        """
        Extract key-value pairs from text (format: "key: value").
        
        Args:
            text: Text with key-value pairs
            
        Returns:
            Dictionary of key-value pairs
        """
        import re
        pattern = r'([^:\n]+):\s*([^\n]+)'
        pairs = re.findall(pattern, text)
        result = {key.strip(): value.strip() for key, value in pairs}
        logger.info(f"Parsed {len(result)} key-value pairs")
        return result
    
    def structure_list_items(self, text: str) -> List[str]:
        """
        Extract list items from text (numbered or bulleted).
        
        Args:
            text: Text containing list items
            
        Returns:
            List of items
        """
        import re
        # Match numbered lists (1. 2. etc) or bullet points (-, *, •)
        pattern = r'(?:^|\n)\s*(?:\d+\.|\-|\*|•)\s*([^\n]+)'
        items = re.findall(pattern, text)
        logger.info(f"Extracted {len(items)} list items")
        return [item.strip() for item in items]


# Convenience functions for easy integration with CrewAI
def save_to_file(content: str, filename: str, format: str = "txt") -> str:
    """
    Quick function to save content to a file.
    Returns: File path or error message.
    """
    writer = FileWriterTool()
    
    if format == "json":
        try:
            data = json.loads(content)
            result = writer.write_json(data, filename)
        except json.JSONDecodeError:
            return "Error: Content is not valid JSON"
    elif format == "md":
        result = writer.write_markdown(content, filename)
    else:
        result = writer.write_text(content, filename)
    
    if result['success']:
        return f"Saved to: {result['file_path']}"
    else:
        return f"Error: {result['error']}"


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """
    Quick function to create a text summary.
    Returns: Summary string.
    """
    summarizer = SummaryTool()
    result = summarizer.create_summary(text, max_sentences=max_sentences)
    
    if result['success']:
        return result['summary']
    else:
        return f"Error: {result['error']}"


def extract_data(text: str, data_type: str = "urls") -> Union[List, Dict]:
    """
    Quick function to extract data from text.
    
    Args:
        text: Text to extract from
        data_type: Type of data - "urls", "emails", "numbers", "key_values", "list_items"
    
    Returns: Extracted data
    """
    parser = DataParserTool()
    
    extractors = {
        "urls": parser.extract_urls,
        "emails": parser.extract_emails,
        "numbers": parser.extract_numbers,
        "key_values": parser.parse_key_value_pairs,
        "list_items": parser.structure_list_items
    }
    
    extractor = extractors.get(data_type, parser.extract_urls)
    return extractor(text)


