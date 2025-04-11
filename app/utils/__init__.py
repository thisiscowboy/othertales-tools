"""
Utility functions for the application.
"""

from app.serper.utils.markdown import HtmlToMarkdown

def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown
    
    Args:
        html: HTML content to convert
        
    Returns:
        Markdown formatted string
    """
    converter = HtmlToMarkdown()
    return converter.convert(html)