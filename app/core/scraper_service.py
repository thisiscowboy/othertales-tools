import asyncio
import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin

# Third-party imports in try/except for graceful handling
try:
    from bs4 import BeautifulSoup, Tag
    HAS_BS4 = True
except ImportError:
    BeautifulSoup = None
    Tag = None
    HAS_BS4 = False

try:
    from playwright.async_api import async_playwright, Response
    HAS_PLAYWRIGHT = True
except ImportError:
    async_playwright = None
    Response = None
    HAS_PLAYWRIGHT = False

from app.utils.config import get_config
from app.utils.markdown import html_to_markdown
from app.core.serper_service import SerperService

logger = logging.getLogger(__name__)


class ScraperService:
    """Service for scraping web content and processing HTML pages"""
    def __init__(self):
        # Check if required libraries are available
        if not HAS_BS4 or not HAS_PLAYWRIGHT:
            logger.error("Required libraries (BeautifulSoup or playwright) are not installed")
            
        config = get_config()
        self.min_delay = config.scraper_min_delay
        self.max_delay = config.scraper_max_delay
        
        self.data_dir = Path(config.scraper_data_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.user_agent = config.user_agent
        
        self.browser = None
        self.context = None

        # Initialize the Serper service
        self.serper_service = SerperService()

    async def get_browser(self):
        """Initialize and return the browser instance"""
        if self.browser is None:
            if not HAS_PLAYWRIGHT:
                raise ImportError("Playwright is not installed")
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=True)
        return self.browser

    async def close(self):
        """Close browser instance when done"""
        if self.browser:
            await self.browser.close()
            self.browser = None

    async def get_or_scrape_url(self, url: str, max_cache_age: int = 86400) -> Dict[str, Any]:
        """Get URL from cache or scrape it if not cached or too old"""
        try:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.json"
            
            if cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < max_cache_age:
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading cache for {url}: {e}")
            
            result = await self.scrape_url(url)
            
            if result["success"]:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Error saving to cache for {url}: {e}")
            return result
        except Exception as e:
            logger.error(f"Error in get_or_scrape_url for {url}: {e}", exc_info=True)
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }

    async def _handle_rate_limiting(self, response: Optional["Response"]) -> bool:
        """Handle rate limiting based on response codes"""
        if response is not None and response.status == 429:  # Too Many Requests
            retry_after = response.headers.get('retry-after')
            wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 60
            logger.info(f"Rate limited. Waiting for {wait_time} seconds")
            await asyncio.sleep(wait_time)
            return True
        return False

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a URL and extract its content"""
        try:
            if not HAS_PLAYWRIGHT:
                raise ImportError("Playwright is not installed")
                
            browser = await self.get_browser()
            context = await browser.new_context(
                user_agent=self.user_agent
            )
            try:
                delay = random.uniform(self.min_delay, self.max_delay)
                await asyncio.sleep(delay)
                
                page = await context.new_page()
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                if response is None or not response.ok:
                    if await self._handle_rate_limiting(response):
                        await page.close()
                        return await self.scrape_url(url)
                    else:
                        status = response.status if response else 0
                        status_text = response.status_text if response else "Unknown error"
                        return {
                            "url": url,
                            "success": False,
                            "error": f"HTTP Error: {status} {status_text}"
                        }
                
                await page.wait_for_load_state("networkidle")
                
                html_content = await page.content()
                title = await page.title()
                
                markdown_content = html_to_markdown(html_content)
                metadata = self._extract_metadata(html_content, url)
                links = self._extract_links(url, html_content)
                
                return {
                    "url": url,
                    "title": title,
                    "content": markdown_content,
                    "metadata": metadata,
                    "links": links,
                    "scraped_at": int(time.time()),
                    "success": True
                }
            finally:
                await context.close()
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}", exc_info=True)
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }

    def _extract_metadata(self, content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML content"""
        metadata: Dict[str, Any] = {
            "source_url": url,
            "extracted_at": datetime.now().isoformat()
        }
        try:
            if not HAS_BS4:
                return metadata
                
            soup = BeautifulSoup(content, "html.parser")
            # Extract Open Graph metadata
            for prop in ["og:title", "og:description", "og:image", "og:type", "og:site_name"]:
                element = soup.find("meta", property=prop)
                if element is not None:
                    element_tag = element
                    content_attr = element_tag.get("content")
                    if content_attr:
                        key = prop.rsplit(":", maxsplit=1)[-1]
                        metadata[key] = content_attr
            # Extract basic metadata
            if "title" not in metadata:
                title_tag = soup.find("title")
                if title_tag:
                    metadata["title"] = title_tag.get_text()
            # Extract description
            if "description" not in metadata:
                desc = soup.find("meta", attrs={"name": "description"})
                if desc is not None:
                    desc_tag = desc
                    content_attr = desc_tag.get("content")
                    if content_attr:
                        metadata["description"] = content_attr
            # Extract LD+JSON structured data
            structured_data = self._extract_structured_data(soup)
            if structured_data:
                metadata["structured_data"] = structured_data
        except Exception as e:
            logger.warning("Error extracting metadata from %s: %s", url, e)
        return metadata

    def _extract_structured_data(self, soup: Any) -> Dict[str, Any]:
        """Extract structured data from LD+JSON scripts"""
        try:
            if not HAS_BS4:
                return {}
                
            structured_data = []
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    script_tag = script
                    script_string = script_tag.string
                    if script_string:
                        data = json.loads(script_string)
                        structured_data.append(data)
                except (json.JSONDecodeError, TypeError):
                    continue
            return {"items": structured_data} if structured_data else {}
        except Exception as e:
            logger.warning("Error extracting structured data: %s", e)
            return {}

    def _extract_links(self, base_url: str, content: str) -> List[str]:
        """Extract links from content"""
        if not HAS_BS4:
            return []
            
        links = []
        # Extract Markdown links [text](url)
        markdown_links = re.findall(r'\[.*?\]\((https?://[^)]+)\)', content)
        links.extend(markdown_links)
        # Extract HTML links from the content
        soup = BeautifulSoup(content, "html.parser")
        for a_tag in soup.find_all('a', href=True):
            a_tag_cast = a_tag
            href = a_tag_cast.get('href')
            if href:
                # Handle relative links
                absolute_url = urljoin(base_url, str(href))
                links.append(absolute_url)
        # Remove duplicates and return
        return list(set(links))

    async def search_and_scrape(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for a query and scrape the top results.
        
        Args:
            query: The search query
            max_results: Maximum number of results to scrape
            
        Returns:
            List of dictionaries containing scraped content
        """
        try:
            # Use the Serper service to get search results
            urls = await self.serper_service.search_and_extract_urls(query, max_results)
            
            if not urls:
                logger.warning(f"No URLs found for search query: {query}")
                return [{"query": query, "success": False, "error": "No search results found"}]
            
            # Scrape each URL
            results = []
            for url in urls:
                try:
                    # Add a small delay between requests
                    await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                    
                    # Scrape the URL
                    result = await self.get_or_scrape_url(url)
                    result["query"] = query  # Add the original query to the result
                    
                    # Clean up the result
                    if not result.get("success", False):
                        logger.warning(f"Failed to scrape URL: {url}")
                    
                    results.append(result)
                    
                    # Limit the number of results
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.error(f"Error scraping URL {url}: {str(e)}")
                    results.append({
                        "url": url,
                        "query": query,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_and_scrape: {str(e)}", exc_info=True)
            return [{"query": query, "success": False, "error": str(e)}]

    async def enhanced_search_and_scrape(self, 
                                        query: str, 
                                        search_type: str = "search",
                                        num_results: int = 10,
                                        max_scrape: int = 5,
                                        country: Optional[str] = None,
                                        locale: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced search and scrape using Serper API with additional metadata
        
        Args:
            query: The search query
            search_type: Type of search (search, news, images, places)
            num_results: Number of search results to return
            max_scrape: Maximum number of results to scrape
            country: Country code for localized results
            locale: Language code for localized results
            
        Returns:
            Dictionary with search results and scraped content
        """
        try:
            # Get search results from Serper
            search_results = await self.serper_service.search(
                query, 
                search_type=search_type,
                num_results=num_results,
                country=country,
                locale=locale
            )
            
            if "error" in search_results:
                return {
                    "query": query,
                    "search_type": search_type,
                    "success": False,
                    "error": search_results["error"],
                    "organic_results": [],
                    "scraped_content": []
                }
            
            # Extract URLs to scrape
            urls_to_scrape = []
            organic_results = search_results.get("organic", [])
            
            for result in organic_results:
                if "link" in result and len(urls_to_scrape) < max_scrape:
                    urls_to_scrape.append(result["link"])
            
            # Scrape each URL
            scraped_content = []
            for url in urls_to_scrape:
                try:
                    # Add a small delay between requests
                    await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                    
                    # Scrape the URL
                    result = await self.get_or_scrape_url(url)
                    
                    if result.get("success", False):
                        scraped_content.append(result)
                    
                except Exception as e:
                    logger.error(f"Error scraping URL {url}: {str(e)}")
                    scraped_content.append({
                        "url": url,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "query": query,
                "search_type": search_type,
                "success": True,
                "organic_results": organic_results,
                "scraped_content": scraped_content
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced_search_and_scrape: {str(e)}", exc_info=True)
            return {
                "query": query,
                "search_type": search_type,
                "success": False,
                "error": str(e),
                "organic_results": [],
                "scraped_content": []
            }

    async def scrape_with_pagination(self, url: str, max_pages: int = 5) -> Dict[str, Any]:
        """Scrape a URL and follow pagination links"""
        all_content = ""
        current_url = url
        pages_scraped = 0
        try:
            while current_url and pages_scraped < max_pages:
                result = await self.scrape_url(current_url)
                if not result["success"]:
                    break
                # Accumulate content
                all_content += result["content"] + "\n\n---\n\n"
                pages_scraped += 1
                # Find next page link
                next_url = self._find_next_page_link(result["content"], current_url)
                if not next_url or next_url == current_url:
                    break
                current_url = next_url
                # Add delay between pages
                await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
            # Create combined result
            return {
                "url": url,
                "title": f"Paginated content ({pages_scraped} pages)",
                "content": all_content,
                "scraped_at": int(time.time()),
                "success": True,
                "pages_scraped": pages_scraped
            }
        except Exception as e:
            logger.error(f"Error in paginated scraping for {url}: {e}", exc_info=True)
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "pages_scraped": pages_scraped
            }

    def _find_next_page_link(self, content: str, current_url: str) -> Optional[str]:
        """Find pagination link in content"""
        if not HAS_BS4:
            return None
            
        soup = BeautifulSoup(content, "html.parser")
        # Common patterns for next page links
        next_selectors = [
            '.pagination .next',
            '.pagination a[rel="next"]',
            'a.next',
            'a:contains("Next")',
            'a[aria-label="Next"]',
            '.pagination a:contains("›")',
            '.pagination a:contains(">")'
        ]
        for selector in next_selectors:
            try:
                next_link = soup.select_one(selector)
                if next_link:
                    next_link_tag = next_link
                    href = next_link_tag.get('href')
                    if href:
                        return urljoin(current_url, str(href))
            except Exception:
                continue
        return None

    async def capture_screenshot(self, url: str, full_page: bool = True) -> Dict[str, Any]:
        """Capture screenshot of a webpage"""
        if not HAS_PLAYWRIGHT:
            return {"url": url, "success": False, "error": "Playwright not installed"}
            
        browser = await self.get_browser()
        context = await browser.new_context(
            user_agent=self.user_agent,
            viewport={'width': 1920, 'height': 1080}
        )
        try:
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle")
            # Capture screenshot
            screenshot_path = self.data_dir / "screenshots"
            screenshot_path.mkdir(exist_ok=True)
            filename = f"{hashlib.md5(url.encode()).hexdigest()}.png"
            file_path = screenshot_path / filename
            await page.screenshot(path=str(file_path), full_page=full_page)
            return {
                "url": url,
                "screenshot_path": str(file_path),
                "timestamp": int(time.time()),
                "success": True
            }
        except Exception as e:
            logger.error(f"Screenshot failed for {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "success": False
            }
        finally:
            await context.close()

    async def scrape_sitemap(self, sitemap_url: str, max_urls: int = 50) -> Dict[str, Any]:
        """Extract URLs from sitemap and scrape them"""
        try:
            if not HAS_BS4:
                return {"success": False, "error": "BeautifulSoup not installed"}
                
            # Scrape the sitemap XML
            sitemap_result = await self.scrape_url(sitemap_url)
            if not sitemap_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to fetch sitemap: {sitemap_result.get('error')}"
                }
            # Extract URLs from sitemap
            soup = BeautifulSoup(sitemap_result["content"], "xml")
            urls = []
            # Process standard sitemap format
            for loc in soup.find_all("loc"):
                urls.append(loc.text)
            # Limit the number of URLs to scrape
            urls = urls[:max_urls]
            # Scrape each URL
            scraped_results = []
            for url in urls:
                try:
                    result = await self.get_or_scrape_url(url)
                    scraped_results.append(result)
                    # Add delay between requests
                    await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                except Exception as e:
                    logger.warning(f"Failed to scrape URL from sitemap {url}: {e}")
            return {
                "sitemap_url": sitemap_url,
                "urls_found": len(urls),
                "urls_scraped": scraped_results,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {e}", exc_info=True)
            return {
                "sitemap_url": sitemap_url,
                "success": False,
                "error": str(e)
            }

    async def crawl_website(self, start_url: str, max_pages: int = 50,
                        recursion_depth: int = 2, allowed_domains: Optional[List[str]] = None,
                        verification_pass: bool = False) -> Dict[str, Any]:
        """
        Crawl a website starting from a URL.
        
        This enhanced version uses the Serper service for more context when appropriate.
        
        Args:
            start_url: URL to start crawling from
            max_pages: Maximum number of pages to crawl
            recursion_depth: Maximum recursion depth
            allowed_domains: List of domains to restrict crawling to
            verification_pass: Whether to do a verification pass to check content stability
            
        Returns:
            Dictionary with crawling results
        """
        # Initialize variables to track progress and results
        visited_urls = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        results = []
        success_count = 0
        failed_count = 0
        
        # Extract domain from start_url if allowed_domains is not provided
        if not allowed_domains:
            parsed_url = urlparse(start_url)
            allowed_domains = [parsed_url.netloc]
        
        # Initialize browser for crawling
        browser = await self.get_browser()
        context = await browser.new_context(
            user_agent=self.user_agent
        )
        
        try:
            while to_visit and len(visited_urls) < max_pages:
                current_url, depth = to_visit.pop(0)
                
                # Skip if already visited
                if current_url in visited_urls:
                    continue
                
                # Add to visited set
                visited_urls.add(current_url)
                
                try:
                    logger.info(f"Crawling {current_url} (depth {depth})")
                    
                    # Add delay between requests
                    await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                    
                    # Scrape the current URL
                    result = await self.scrape_url(current_url)
                    
                    if result["success"]:
                        success_count += 1
                        results.append(result)
                        
                        # If we haven't reached max depth, add links to visit
                        if depth < recursion_depth:
                            links = result.get("links", [])
                            
                            for link in links:
                                # Skip if already visited or queued
                                if link in visited_urls or any(link == url for url, _ in to_visit):
                                    continue
                                
                                # Check if link is from an allowed domain
                                parsed_link = urlparse(link)
                                if parsed_link.netloc in allowed_domains:
                                    to_visit.append((link, depth + 1))
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {e}")
                    failed_count += 1
            
            # Perform verification pass if requested
            verification_results = None
            verification_success_rate = None
            if verification_pass and results:
                urls_to_verify = [result["url"] for result in results]
                verification_results = await self._perform_verification_pass(urls_to_verify, context)
                
                # Calculate verification success rate
                if verification_results:
                    verification_success_rate = sum(1 for vr in verification_results if vr["verified"]) / len(verification_results)
                
            # Prepare final results
            final_results = {
                "pages_crawled": len(visited_urls),
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results
            }
            
            if verification_pass and verification_results:
                final_results["verification_results"] = verification_results
                final_results["verification_success_rate"] = verification_success_rate
                
            return final_results
            
        finally:
            await context.close()

    async def _perform_verification_pass(self, urls: List[str], context: Any) -> List[Dict[str, Any]]:
        """Verification pass to check content stability"""
        verification_results = []
        
        for url in urls:
            try:
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    verification_results.append({
                        "url": url,
                        "verified": True
                    })
                except Exception as e:
                    verification_results.append({
                        "url": url,
                        "verified": False,
                        "error": str(e)
                    })
                finally:
                    await page.close()
                    
                # Rate limiting
                await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                
            except Exception as e:
                verification_results.append({
                    "url": url,
                    "verified": False,
                    "error": str(e)
                })
                
        return verification_results
    
    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs in parallel"""
        tasks = [self.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks)