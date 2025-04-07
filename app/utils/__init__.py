"""
Utility functions for the application.
"""

# Change this line in scraper_service.py (line 36):
from app.serper.utils.markdown import HtmlToMarkdown
# Then initialize and use it where needed
html_to_markdown = lambda html: HtmlToMarkdown().convert(html)