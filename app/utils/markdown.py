from bs4 import BeautifulSoup, Comment, NavigableString, Tag
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Clean up text by removing extra whitespace and normalizing line breaks"""
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def _is_navigation_element(tag):
    """Identify if an element is likely navigation or template content"""
    if not tag or not isinstance(tag, Tag):
        return False
    # Check class names
    nav_classes = ["nav", "menu", "navigation", "banner", "header", "footer", "sidebar", "cookie"]
    if tag.has_attr("class"):
        for cls in tag.get("class", []):
            if any(nav_term in cls.lower() for nav_term in nav_classes):
                return True
    # Check element ID
    if tag.has_attr("id"):
        if any(nav_term in tag.get("id", "").lower() for nav_term in nav_classes):
            return True
    # Check role attribute
    if tag.has_attr("role"):
        nav_roles = ["navigation", "banner", "menu", "menubar", "complementary"]
        if tag.get("role", "").lower() in nav_roles:
            return True
    return False


def calculate_readability_score(soup: BeautifulSoup) -> float:
    """Calculate a readability score for the content"""
    # Simple implementation - can be expanded with readability algorithms
    text = soup.get_text()
    words = len(re.findall(r"\b\w+\b", text))
    sentences = len(re.findall(r"[.!?]+", text)) or 1
    # Calculate average words per sentence (basic readability metric)
    avg_words_per_sentence = words / sentences
    # Detect if text is likely substantive content
    paragraphs = len(soup.find_all("p"))
    has_headings = len(soup.find_all(["h1", "h2", "h3"])) > 0
    # Score based on several factors (0-100)
    score = 50  # Base score
    # Adjust based on sentence length (penalize very short and very long sentences)
    if 10 <= avg_words_per_sentence <= 25:
        score += 20
    elif avg_words_per_sentence > 25:
        score -= 10
    elif avg_words_per_sentence < 5:
        score -= 10
    # Adjust based on content structure
    if paragraphs > 3:
        score += 15
    if has_headings:
        score += 15
    # Normalize to 0-1 range
    return min(100, max(0, score)) / 100


def extract_main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Extract the main content area from a page"""
    # Look for common content containers
    content_ids = ["content", "main", "article", "post", "entry"]
    content_classes = ["content", "article", "post", "entry", "main"]
    # Try to find content by ID
    for id_name in content_ids:
        content = soup.find(id=id_name)
        if content:
            return content
    # Try to find by class
    for class_name in content_classes:
        content = soup.find(class_=class_name)
        if content:
            return content
    # Look for article tag
    article = soup.find("article")
    if article:
        return article
    # Look for main tag
    main = soup.find("main")
    if main:
        return main
    # Try to find the element with the most paragraphs
    paragraphs_by_container = {}
    for container in soup.find_all(["div", "section"]):
        paragraphs = len(container.find_all("p"))
        if paragraphs > 2:  # Only consider containers with multiple paragraphs
            paragraphs_by_container[container] = paragraphs
    if paragraphs_by_container:
        main_container = max(paragraphs_by_container, key=paragraphs_by_container.get)
        return main_container
    # If no content container found, use the body
    return soup.body or soup


def html_to_markdown(html_content: str, url: str = "", title: str = "") -> str:
    """Convert HTML to clean Markdown format with improved content extraction"""
    try:
        soup = BeautifulSoup(html_content, "lxml")
        # Add content readability score
        readability_score = calculate_readability_score(soup)
        # Improved content extraction based on readability metrics
        main_content = extract_main_content(soup)
        # Clean up the HTML
        for element in soup.find_all(["script", "style", "iframe", "noscript"]):
            element.decompose()
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        # Remove likely navigation elements
        for element in soup.find_all():
            if _is_navigation_element(element):
                element.decompose()
        # Extract title if not provided
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        # Start with title as heading
        markdown = f"# {title}\n\n"
        if url:
            markdown += f"URL Source: {url}\n\n"
        if main_content:
            # Process headings
            for heading in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                level = int(heading.name[1])
                text = _clean_text(heading.get_text())
                if text:
                    markdown += f"\n{'#' * level} {text}\n\n"
            # Process paragraphs
            for p in main_content.find_all("p"):
                text = _clean_text(p.get_text())
                if text:
                    markdown += f"{text}\n\n"
            # Process unordered lists
            for ul in main_content.find_all("ul"):
                for li in ul.find_all("li", recursive=False):
                    text = _clean_text(li.get_text())
                    if text:
                        markdown += f"* {text}\n"
                markdown += "\n"
            # Process ordered lists
            for ol in main_content.find_all("ol"):
                for i, li in enumerate(ol.find_all("li", recursive=False), 1):
                    text = _clean_text(li.get_text())
                    if text:
                        markdown += f"{i}. {text}\n"
                markdown += "\n"
            # Process blockquotes
            for blockquote in main_content.find_all("blockquote"):
                text = _clean_text(blockquote.get_text())
                if text:
                    lines = text.splitlines()
                    for line in lines:
                        markdown += f"> {line}\n"
                    markdown += "\n"
            # Process tables
            for table in main_content.find_all("table"):
                markdown += _table_to_markdown(table) + "\n\n"
            # Process code blocks
            for pre in main_content.find_all("pre"):
                code = pre.get_text()
                markdown += f"```\n{code}\n```\n\n"
            # Process links
            for link in main_content.find_all("a", href=True):
                href = link.get("href")
                text = _clean_text(link.get_text())
                if (
                    text
                    and href
                    and not href.startswith("#")
                    and not href.startswith("javascript:")
                ):
                    # Use reference style links to keep the text clean
                    markdown = markdown.replace(text, f"[{text}]({href})")
        # Clean up excess whitespace and normalize line breaks
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        # Add metadata about extraction quality
        metadata_section = (
            f"\n\n---\n*Content extracted with {int(readability_score * 100)}% confidence*"
        )
        markdown += metadata_section
        return markdown
    except Exception as e:
        logger.error(f"Error converting HTML to Markdown: {e}")
        # Return basic markdown if conversion fails
        return f"# {title}\n\nURL Source: {url}\n\n[Content extraction failed]"


def _table_to_markdown(table_tag) -> str:
    """Convert HTML table to Markdown table"""
    rows = []
    # Extract headers
    headers = []
    header_row = table_tag.find("thead")
    if header_row:
        th_elements = header_row.find_all(["th", "td"])
        headers = [_clean_text(th.get_text()) for th in th_elements]
    # If no headers found in thead, try the first tr
    if not headers:
        first_row = table_tag.find("tr")
        if first_row:
            th_elements = first_row.find_all(["th", "td"])
            headers = [_clean_text(th.get_text()) for th in th_elements]
    # Process the table body
    tbody = table_tag.find("tbody") or table_tag
    data_rows = []
    # Skip the first row if we already used it for headers
    start_idx = 1 if not table_tag.find("thead") and headers else 0
    for tr in tbody.find_all("tr")[start_idx:]:
        cells = [_clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
        if any(cells):  # Skip empty rows
            data_rows.append(cells)
    # Create the markdown table
    markdown_table = ""
    # Add headers if we have them
    if headers:
        markdown_table += "| " + " | ".join(headers) + " |\n"
        markdown_table += "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|\n"
    # Add data rows
    for row in data_rows:
        # Pad the row if necessary to match header length
        if headers and len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        markdown_table += "| " + " | ".join(row) + " |\n"
    return markdown_table


def extract_structured_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract structured data from HTML page"""
    structured_data = {}
    # Extract JSON-LD
    json_ld_data = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                json_ld_data.append(data)
            elif isinstance(data, list):
                json_ld_data.extend(data)
        except Exception as e:
            logger.error(f"Error parsing JSON-LD: {e}")
    if json_ld_data:
        structured_data["json_ld"] = json_ld_data
    # Extract microdata
    microdata = {}
    for element in soup.find_all(itemscope=True):
        if element.has_attr("itemtype"):
            item_type = element["itemtype"]
            item_props = {}
            # Extract properties
            for prop in element.find_all(itemprop=True):
                prop_name = prop["itemprop"]
                # Get property value based on tag type
                if prop.name == "meta":
                    prop_value = prop.get("content", "")
                elif prop.name == "img":
                    prop_value = prop.get("src", "")
                elif prop.name == "a":
                    prop_value = prop.get("href", "")
                elif prop.name == "time":
                    prop_value = prop.get("datetime", prop.get_text())
                else:
                    prop_value = prop.get_text().strip()
                item_props[prop_name] = prop_value
            if item_type not in microdata:
                microdata[item_type] = []
            microdata[item_type].append(item_props)
    if microdata:
        structured_data["microdata"] = microdata
    # Extract OpenGraph metadata
    og_data = {}
    for meta in soup.find_all("meta", property=re.compile(r"^og:")):
        prop = meta.get("property", "").replace("og:", "")
        content = meta.get("content", "")
        if prop and content:
            og_data[prop] = content
    if og_data:
        structured_data["opengraph"] = og_data
    # Extract Twitter card metadata
    twitter_data = {}
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
        prop = meta.get("name", "").replace("twitter:", "")
        content = meta.get("content", "")
        if prop and content:
            twitter_data[prop] = content
    if twitter_data:
        structured_data["twitter_card"] = twitter_data
    return structured_data
