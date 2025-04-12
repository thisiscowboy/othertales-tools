import asyncio
import hashlib
import json
import logging
import random
import re
import time
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
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
    from playwright.async_api import async_playwright, Response, Page, Browser, BrowserContext
    HAS_PLAYWRIGHT = True
except ImportError:
    async_playwright = None
    Response = None
    Page = None
    Browser = None
    BrowserContext = None
    HAS_PLAYWRIGHT = False

# GitPython import
try:
    import git
    HAS_GIT = True
except ImportError:
    git = None
    HAS_GIT = False

from app.utils.config import get_config
from app.utils.markdown import html_to_markdown
from app.core.serper_service import SerperService
from app.core.git_service import GitService

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
        
        # Create repos directory for GitHub repositories
        self.repos_dir = self.data_dir / "repos"
        self.repos_dir.mkdir(exist_ok=True)
        
        # Use the user agent from config
        self.user_agent = config.user_agent
        
        # Initialize Serper service for search
        self.serper_service = SerperService()
        
        # Initialize Git service
        self.git_service = GitService()
        
        # Playwright browser reference
        self._browser = None
        self._browser_context = None
        self._browser_lock = asyncio.Lock()
        self._active_tasks = set()  # Track active async tasks for proper cleanup

    async def get_browser(self) -> "Browser":
        """Get or create a browser instance in a thread-safe manner"""
        async with self._browser_lock:
            if self._browser is None:
                try:
                    logger.info("Initializing browser")
                    self._playwright = await async_playwright().start()
                    # Use chromium for better compatibility
                    self._browser = await self._playwright.chromium.launch(headless=True)
                except Exception as e:
                    logger.error(f"Failed to initialize browser: {e}")
                    # Reset state in case of failure
                    self._browser = None
                    self._playwright = None
                    raise
            return self._browser
            
    async def close_browser(self):
        """Safely close the browser and playwright instance"""
        async with self._browser_lock:
            if self._browser:
                try:
                    await self._browser.close()
                except Exception as e:
                    logger.error(f"Error closing browser: {e}")
                finally:
                    self._browser = None
                    
            if hasattr(self, '_playwright') and self._playwright:
                try:
                    await self._playwright.stop()
                except Exception as e:
                    logger.error(f"Error stopping playwright: {e}")
                finally:
                    self._playwright = None
    
    async def get_or_scrape_url(self, url: str) -> Dict[str, Any]:
        """Get cached content or scrape URL if not cached"""
        try:
            # Generate cache key from URL
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Check if we have a recent cache
            if cache_file.exists():
                try:
                    cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                    cache_timestamp = cached_data.get("scraped_at", 0)
                    # Use cache if it's less than 24 hours old
                    if time.time() - cache_timestamp < 86400:
                        logger.info(f"Using cached content for {url}")
                        return cached_data
                except Exception as e:
                    logger.warning(f"Failed to load cache for {url}: {e}")
            
            # Scrape the URL and update cache
            result = await self.scrape_url(url)
            
            # Only cache successful scrapes
            if result["success"]:
                cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                
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

    async def scrape_url(self, url: str, wait_for_selector: Optional[str] = None, 
                         wait_timeout: int = 30000, extract_tables: bool = True,
                         handle_javascript: bool = True) -> Dict[str, Any]:
        """
        Scrape a URL and extract its content
        
        Args:
            url: The URL to scrape
            wait_for_selector: Optional CSS selector to wait for
            wait_timeout: Timeout for page load in milliseconds
            extract_tables: Whether to extract tables as structured data
            handle_javascript: Whether to process JavaScript on the page
            
        Returns:
            Dictionary with scraped content
        """
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright is not installed")
            
        browser = await self.get_browser()
        context = await browser.new_context(user_agent=self.user_agent)
        
        try:
            # Add random delay between requests
            delay = random.uniform(self.min_delay, self.max_delay)
            await asyncio.sleep(delay)
            
            page = await context.new_page()
            
            # Set up extra request handlers for specific content types
            content_type = "text/html"  # Default content type
            
            response = await page.goto(url, wait_until="domcontentloaded", timeout=wait_timeout)
            
            if response is None or not response.ok:
                if await self._handle_rate_limiting(response):
                    await page.close()
                    return await self.scrape_url(url, wait_for_selector, wait_timeout, extract_tables)
                
                status = response.status if response else "Unknown"
                return {
                    "url": url,
                    "success": False,
                    "error": f"Failed to load page (status: {status})"
                }
            
            # Get the content type from response
            content_type = response.headers.get("content-type", "text/html")
            
            # Wait for specified selector or additional time for JavaScript to execute
            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=wait_timeout)
                except Exception as e:
                    logger.warning(f"Selector {wait_for_selector} not found: {e}")
            else:
                # Give the page some time to fully render
                await asyncio.sleep(2)
            
            # Handle different content types
            if "application/pdf" in content_type:
                # For PDFs, extract text if possible or return a placeholder
                return await self._handle_pdf_content(page, url, response)
            elif "text/html" in content_type:
                # For HTML, process as usual
                return await self._process_html_page(page, url, extract_tables)
            else:
                # For other content types, try to get raw content
                logger.warning(f"Unsupported content type: {content_type} for {url}")
                return {
                    "url": url,
                    "title": url.split("/")[-1],
                    "content": f"Unsupported content type: {content_type}",
                    "content_type": content_type,
                    "success": False,
                    "error": f"Unsupported content type: {content_type}"
                }
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}", exc_info=True)
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }
        finally:
            await context.close()
    
    async def _handle_pdf_content(self, page: "Page", url: str, response: "Response") -> Dict[str, Any]:
        """Handle PDF content extraction"""
        try:
            # Get the PDF filename from the URL or Content-Disposition header
            filename = url.split("/")[-1]
            if not filename.lower().endswith(".pdf"):
                filename = f"document_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
            
            # For now, just return a reference to the PDF
            # In a future version, this could be enhanced with PDF text extraction
            return {
                "url": url,
                "title": f"PDF: {filename}",
                "content": f"PDF Document available at: {url}\n\nFilename: {filename}",
                "content_type": "application/pdf",
                "links": [],
                "metadata": {
                    "content_type": "application/pdf",
                    "filename": filename,
                    "source_url": url
                },
                "scraped_at": int(time.time()),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing PDF {url}: {e}", exc_info=True)
            return {
                "url": url,
                "success": False,
                "error": f"PDF processing error: {str(e)}"
            }
            
    async def _process_html_page(self, page: "Page", url: str, extract_tables: bool) -> Dict[str, Any]:
        """Process an HTML page and extract content"""
        # Get the page title
        title = await page.title()
        
        # Get the HTML content
        html_content = await page.content()
        
        # Extract relevant info using BeautifulSoup
        metadata = {}
        links = []
        tables = []
        
        if html_content and HAS_BS4:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract metadata
            for meta in soup.find_all("meta"):
                if isinstance(meta, Tag):
                    if meta.get("name") and meta.get("content"):
                        metadata[meta["name"]] = meta["content"]
                    elif meta.get("property") and meta.get("content"):
                        metadata[meta["property"]] = meta["content"]
            
            # Extract links
            for a_tag in soup.find_all("a", href=True):
                if isinstance(a_tag, Tag):
                    href = a_tag["href"]
                    try:
                        # Make absolute URL
                        absolute_url = urljoin(url, href)
                        parsed = urlparse(absolute_url)
                        
                        # Only include HTTP/HTTPS URLs (exclude JavaScript, mailto, etc.)
                        if parsed.scheme in ("http", "https") and parsed.netloc:
                            links.append(absolute_url)
                    except Exception:
                        pass
            
            # Extract tables if requested
            if extract_tables:
                for i, table in enumerate(soup.find_all("table")):
                    if isinstance(table, Tag):
                        try:
                            table_data = {"headers": [], "rows": []}
                            
                            # Extract headers
                            headers = []
                            for th in table.find_all("th"):
                                if isinstance(th, Tag):
                                    headers.append(th.get_text(strip=True))
                            
                            # If no headers found in th, try first tr
                            if not headers and table.find("tr"):
                                first_row = table.find("tr")
                                if isinstance(first_row, Tag):
                                    for td in first_row.find_all("td"):
                                        if isinstance(td, Tag):
                                            headers.append(td.get_text(strip=True))
                            
                            table_data["headers"] = headers
                            
                            # Extract rows
                            rows = []
                            for tr in table.find_all("tr")[1:] if headers else table.find_all("tr"):
                                if isinstance(tr, Tag):
                                    row = []
                                    for td in tr.find_all(["td", "th"]):
                                        if isinstance(td, Tag):
                                            row.append(td.get_text(strip=True))
                                    if row:
                                        rows.append(row)
                            
                            table_data["rows"] = rows
                            
                            if headers or rows:
                                tables.append(table_data)
                        except Exception as e:
                            logger.warning(f"Error extracting table {i} from {url}: {e}")
            
            # Convert HTML to Markdown
            markdown_content = html_to_markdown(html_content)
        else:
            # Fallback if BeautifulSoup is not available
            markdown_content = f"# {title}\n\nContent for {url} could not be processed."
        
        # Return the structured result
        return {
            "url": url,
            "title": title,
            "content": markdown_content,
            "links": links,
            "metadata": metadata,
            "tables": tables if tables else None,
            "domain": urlparse(url).netloc,
            "scraped_at": int(time.time()),
            "success": True
        }

    async def scrape_sitemap(self, sitemap_url: str, max_urls: int = 500) -> Dict[str, Any]:
        """
        Extract URLs from sitemap and scrape them
        
        Enhanced to handle large sitemaps and sitemap indexes
        """
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
            
            # Check if this is a sitemap index
            is_sitemap_index = len(soup.find_all("sitemapindex")) > 0
            
            if is_sitemap_index:
                logger.info(f"Found sitemap index at {sitemap_url}")
                # Process sitemap index - collect child sitemaps
                child_sitemaps = []
                for sitemap_tag in soup.find_all("sitemap"):
                    if isinstance(sitemap_tag, Tag):
                        loc_tag = sitemap_tag.find("loc")
                        if isinstance(loc_tag, Tag) and loc_tag.string:
                            child_sitemaps.append(loc_tag.string)
                
                # Process each child sitemap
                for child_sitemap_url in child_sitemaps[:10]:  # Limit to 10 child sitemaps
                    logger.info(f"Processing child sitemap: {child_sitemap_url}")
                    child_result = await self.scrape_sitemap(child_sitemap_url, max_urls // len(child_sitemaps[:10]))
                    if child_result.get("success") and "urls_scraped" in child_result:
                        # Add child sitemap results to our results
                        urls.extend([r["url"] for r in child_result["urls_scraped"] if r.get("success")])
            else:
                # Process standard sitemap
                for loc in soup.find_all("loc"):
                    if loc.string:
                        urls.append(loc.string)
            
            # Limit the number of URLs to scrape
            unique_urls = list(set(urls))  # Remove duplicates
            urls_to_scrape = unique_urls[:max_urls]
            
            # Scrape each URL
            scraped_results = []
            skipped_count = 0
            
            for url in urls_to_scrape:
                try:
                    logger.info(f"Scraping URL from sitemap: {url}")
                    result = await self.get_or_scrape_url(url)
                    scraped_results.append(result)
                    # Add delay between requests
                    await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                except Exception as e:
                    logger.warning(f"Failed to scrape URL from sitemap {url}: {e}")
                    skipped_count += 1
                    
            return {
                "sitemap_url": sitemap_url,
                "urls_found": len(unique_urls),
                "urls_selected": len(urls_to_scrape),
                "urls_scraped": scraped_results,
                "skipped_count": skipped_count,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {e}", exc_info=True)
            return {
                "sitemap_url": sitemap_url,
                "success": False,
                "error": str(e)
            }

    async def crawl_website(self, start_url: str, max_pages: int = 5000,
                        recursion_depth: int = 10, allowed_domains: Optional[List[str]] = None,
                        verification_pass: bool = False, site_type: Optional[str] = None,
                        follow_subdomains: bool = True, content_types: List[str] = ["text/html"],
                        include_patterns: Optional[List[str]] = None,
                        exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Crawl a website starting from a URL.
        
        Enhanced version optimized for large government websites and specialized content.
        
        Args:
            start_url: URL to start crawling from
            max_pages: Maximum number of pages to crawl
            recursion_depth: Maximum recursion depth
            allowed_domains: List of domains to restrict crawling to
            verification_pass: Whether to do a verification pass to check content stability
            site_type: Predefined site type (legislation, hmrc, generic)
            follow_subdomains: Whether to follow links to subdomains of allowed domains
            content_types: Content types to process
            include_patterns: URL patterns to include (regex patterns)
            exclude_patterns: URL patterns to exclude (regex patterns)
            
        Returns:
            Dictionary with crawling results
        """
        # Initialize variables to track progress and results
        visited_urls = set()
        
        # Apply site-specific configurations for known sites
        if site_type == "legislation":
            # Configuration for legislation.gov.uk
            if not allowed_domains:
                allowed_domains = ["legislation.gov.uk"]
            if not include_patterns:
                include_patterns = [r"\/ukpga\/", r"\/uksi\/", r"\/primary\+secondary"]
            if not exclude_patterns:
                exclude_patterns = [r"\/changes\/", r"\/status\/"]
            # Increase depth for legislation site
            recursion_depth = max(recursion_depth, 15)
            
        elif site_type == "hmrc":
            # Configuration for HMRC tax documentation
            if not allowed_domains:
                allowed_domains = ["gov.uk"]
            if not include_patterns:
                include_patterns = [r"\/hmrc\/", r"\/government\/publications", r"\/guidance\/"]
            # Include PDFs for HMRC documentation
            if "application/pdf" not in content_types:
                content_types.append("application/pdf")
            # Increase depth for HMRC site
            recursion_depth = max(recursion_depth, 15)
            
        # Process the domain(s)
        base_domains = [] if allowed_domains is None else allowed_domains.copy()
        
        # Extract domain from start_url
        start_domain = urlparse(start_url).netloc
        if start_domain and start_domain not in base_domains:
            base_domains.append(start_domain)
            
        # Expand to allow subdomains if enabled
        if follow_subdomains:
            domain_set = set()
            for domain in base_domains:
                # Extract the main domain (example.com from sub.example.com)
                parts = domain.split('.')
                if len(parts) > 1:
                    # Get the main domain (last two parts)
                    main_domain = '.'.join(parts[-2:])
                    domain_set.add(main_domain)
                else:
                    domain_set.add(domain)
            allowed_domains = list(domain_set)
        else:
            allowed_domains = base_domains
            
        # Compile regex patterns
        compiled_include_patterns = None
        compiled_exclude_patterns = None
        
        if include_patterns:
            compiled_include_patterns = [re.compile(pattern) for pattern in include_patterns]
            
        if exclude_patterns:
            compiled_exclude_patterns = [re.compile(pattern) for pattern in exclude_patterns]
            
        # Setup crawling queue and results tracking
        to_visit = [(start_url, 0)]  # (url, depth)
        results = []
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Initialize browser for crawling
        browser = await self.get_browser()
        context = await browser.new_context(user_agent=self.user_agent)
        
        try:
            while to_visit and len(visited_urls) < max_pages:
                current_url, depth = to_visit.pop(0)
                
                # Skip if already visited
                if current_url in visited_urls:
                    continue
                    
                # Add to visited set
                visited_urls.add(current_url)
                
                # Check URL against patterns
                should_process = True
                
                # Check exclude patterns first
                if compiled_exclude_patterns:
                    for pattern in compiled_exclude_patterns:
                        if pattern.search(current_url):
                            logger.info(f"Skipping {current_url} - matches exclude pattern")
                            should_process = False
                            skipped_count += 1
                            break
                
                # Then check include patterns
                if should_process and compiled_include_patterns:
                    should_process = False
                    for pattern in compiled_include_patterns:
                        if pattern.search(current_url):
                            should_process = True
                            break
                    if not should_process:
                        logger.info(f"Skipping {current_url} - doesn't match include patterns")
                        skipped_count += 1
                
                # Process the URL if it passes all filters
                if should_process:
                    try:
                        logger.info(f"Crawling {current_url} (depth {depth})")
                        
                        # Add delay between requests
                        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                        
                        # Scrape the current URL
                        result = await self.scrape_url(current_url)
                        
                        if result["success"]:
                            # Add depth information
                            result["depth"] = depth
                            
                            success_count += 1
                            results.append(result)
                            
                            # If we haven't reached max depth, add links to visit
                            if depth < recursion_depth:
                                links = result.get("links", [])
                                
                                for link in links:
                                    # Skip if already visited or queued
                                    if link in visited_urls or any(link == url for url, _ in to_visit):
                                        continue
                                    
                                    # Check domain restrictions
                                    parsed_link = urlparse(link)
                                    link_domain = parsed_link.netloc
                                    
                                    # Domain check based on configuration
                                    if follow_subdomains:
                                        # Check if the link domain is a subdomain of any allowed domain
                                        domain_allowed = False
                                        for allowed_domain in allowed_domains:
                                            if link_domain.endswith(allowed_domain):
                                                domain_allowed = True
                                                break
                                    else:
                                        # Strict domain check
                                        domain_allowed = link_domain in allowed_domains
                                    
                                    if domain_allowed:
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
                "skipped_count": skipped_count,
                "results": results
            }
            
            if verification_results:
                final_results["verification_results"] = verification_results
                final_results["verification_success_rate"] = verification_success_rate
                
            return final_results
            
        except Exception as e:
            logger.error(f"Error during crawl: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "pages_crawled": len(visited_urls),
                "results": results
            }
        finally:
            try:
                await context.close()
            except Exception:
                pass
    
    async def _perform_verification_pass(self, urls: List[str], context: "BrowserContext") -> List[Dict[str, Any]]:
        """
        Perform a verification pass to check content stability and completeness
        
        This enhanced verification:
        1. Re-scrapes each URL to check content stability
        2. Compares content length and structure to ensure completeness
        3. Checks for key elements like headings, links, and content blocks
        4. Identifies potential issues like truncated content or rendering problems
        
        Returns a detailed verification report for each URL
        """
        verification_results = []
        
        for url in urls:
            try:
                # Scrape the URL again
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for the page to stabilize (longer wait for complex pages)
                await asyncio.sleep(3)
                
                # Get the HTML content
                new_content = await page.content()
                
                # Check that page loaded properly
                title = await page.title()
                
                # Extract key content elements to verify completeness
                completeness_checks = {}
                
                # 1. Check for headings
                heading_count = await page.evaluate("""() => {
                    return document.querySelectorAll('h1, h2, h3, h4, h5, h6').length;
                }""")
                completeness_checks["headings"] = heading_count
                
                # 2. Check for content paragraphs
                paragraph_count = await page.evaluate("""() => {
                    return document.querySelectorAll('p').length;
                }""")
                completeness_checks["paragraphs"] = paragraph_count
                
                # 3. Check for links
                link_count = await page.evaluate("""() => {
                    return document.querySelectorAll('a[href]').length;
                }""")
                completeness_checks["links"] = link_count
                
                # 4. Check for images
                image_count = await page.evaluate("""() => {
                    return document.querySelectorAll('img').length;
                }""")
                completeness_checks["images"] = image_count
                
                # 5. Check content size
                content_size = len(new_content)
                completeness_checks["content_size"] = content_size
                
                # Perform content analysis by looking for common page elements
                # Different page types will have different expectations
                completeness_level = "unknown"
                
                if content_size < 1000:
                    completeness_level = "minimal"
                elif heading_count == 0 and paragraph_count < 5:
                    completeness_level = "partial"
                elif heading_count > 0 and paragraph_count > 5:
                    completeness_level = "substantial"
                    if link_count > 5 and content_size > 10000:
                        completeness_level = "complete"
                
                # Take a screenshot for visual verification
                screenshot = None
                try:
                    screenshot_path = self.cache_dir / f"verify_{hashlib.md5(url.encode()).hexdigest()}.png"
                    await page.screenshot(path=str(screenshot_path))
                    screenshot = str(screenshot_path)
                except Exception as screenshot_error:
                    logger.warning(f"Failed to take verification screenshot: {screenshot_error}")
                
                # Close the page
                await page.close()
                
                # Determine verification status
                verified = content_size > 5000 and completeness_level in ["substantial", "complete"]
                
                # Create detailed verification result
                verification_results.append({
                    "url": url,
                    "verified": verified,
                    "title": title,
                    "content_size": content_size,
                    "completeness_level": completeness_level,
                    "completeness_checks": completeness_checks,
                    "screenshot": screenshot,
                    "verification_time": datetime.now().isoformat()
                })
                
                # Add delay between requests
                await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                
            except Exception as e:
                logger.error(f"Error during verification for {url}: {e}")
                verification_results.append({
                    "url": url,
                    "verified": False,
                    "error": str(e),
                    "verification_time": datetime.now().isoformat()
                })
                
        return verification_results
    
    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs in parallel with rate limiting"""
        results = []
        for url in urls:
            try:
                # Add some delay between requests
                await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                result = await self.scrape_url(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })
        return results
    
    async def search_and_scrape(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search and scrape results"""
        try:
            # Use Serper to search
            search_results = await self.serper_service.search(query, max_results)
            
            if not search_results.get("organic"):
                logger.warning(f"No search results found for query: {query}")
                return []
            
            # Extract URLs from search results
            urls = []
            for result in search_results.get("organic", []):
                if "link" in result:
                    urls.append(result["link"])
            
            if not urls:
                logger.warning("No valid URLs found in search results")
                return []
            
            # Scrape each URL
            scraped_results = await self.scrape_urls(urls[:max_results])
            
            # Add query context to results
            for result in scraped_results:
                if result.get("success"):
                    result["metadata"] = result.get("metadata", {})
                    result["metadata"]["search_query"] = query
            
            return scraped_results
            
        except Exception as e:
            logger.error(f"Error in search_and_scrape: {e}", exc_info=True)
            return []
    
    async def scrape_github_repository(
        self, 
        repo_url: str,
        auth_token: Optional[str] = None,
        doc_folders: List[str] = ["docs", "documentation", "doc", "examples", "cookbook"],
        file_extensions: List[str] = [".md", ".rst", ".txt"],
        exclude_patterns: Optional[List[str]] = None,
        extract_code_examples: bool = True,
        max_files: int = 500
    ) -> Dict[str, Any]:
        """
        Scrape documentation and code examples from a GitHub repository
        
        Args:
            repo_url: URL of the GitHub repository
            auth_token: GitHub authentication token for private repos
            doc_folders: List of folders to focus on for documentation
            file_extensions: List of file extensions to include
            exclude_patterns: Patterns to exclude
            extract_code_examples: Whether to extract and process code examples
            max_files: Maximum number of files to process
            
        Returns:
            Dictionary with scraped repository information and content
        """
        if not HAS_GIT:
            return {
                "success": False,
                "error": "GitPython is not installed",
                "repo_url": repo_url
            }
            
        # Create a temporary directory for the clone
        repo_id = hashlib.md5(repo_url.encode()).hexdigest()[:10]
        repo_path = self.repos_dir / repo_id
        
        # Normalize GitHub URL to handle different formats
        github_url = self._normalize_github_url(repo_url)
        if not github_url:
            return {
                "success": False,
                "error": "Invalid GitHub repository URL format",
                "repo_url": repo_url
            }
        
        try:
            # Clean up existing repo directory if it exists
            if repo_path.exists():
                shutil.rmtree(repo_path)
                
            # Clone the repository
            logger.info(f"Cloning repository {github_url} to {repo_path}")
            clone_result = self.git_service.clone_repo(github_url, str(repo_path), auth_token)
            
            # Get repository information
            repo_info = self._extract_repo_info(repo_path)
            
            # Find documentation files
            doc_files = self._find_documentation_files(
                repo_path, 
                doc_folders,
                file_extensions, 
                exclude_patterns
            )
            
            if not doc_files:
                return {
                    "success": True,
                    "repo_url": github_url,
                    "repo_info": repo_info,
                    "warning": "No documentation files found matching criteria",
                    "files_found": 0,
                    "files_processed": 0,
                    "documents": []
                }
            
            # Limit the number of files to process
            files_to_process = doc_files[:max_files]
            
            # Process documentation files
            docs_results = []
            for file_path in files_to_process:
                # Process the file based on its extension
                try:
                    result = self._process_documentation_file(file_path, repo_path, github_url)
                    if result:
                        docs_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
            
            # Extract code examples if requested
            code_examples = []
            if extract_code_examples:
                code_examples = self._extract_code_examples(repo_path, exclude_patterns)
            
            # Clean up temporary files (optional - can keep for caching)
            # shutil.rmtree(repo_path)
            
            return {
                "success": True,
                "repo_url": github_url,
                "repo_info": repo_info,
                "files_found": len(doc_files),
                "files_processed": len(files_to_process),
                "documents": docs_results,
                "code_examples": code_examples
            }
            
        except Exception as e:
            logger.error(f"Error scraping GitHub repository {repo_url}: {e}", exc_info=True)
            # Clean up on error
            if repo_path.exists():
                shutil.rmtree(repo_path)
                
            return {
                "success": False,
                "error": str(e),
                "repo_url": github_url
            }
    
    def _normalize_github_url(self, url: str) -> Optional[str]:
        """Normalize GitHub URL to standard format for cloning"""
        # Try to match GitHub URL patterns
        patterns = [
            # Standard GitHub URL
            r'https?://github\.com/([^/]+)/([^/]+)(?:/.*)?',
            # GitHub enterprise URL
            r'https?://([^/]+)/([^/]+)/([^/]+)(?:/.*)?',
            # SSH URL
            r'git@github\.com:([^/]+)/([^/]+)\.git'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                if 'github.com' in url or 'git@github.com' in url:
                    # Standard GitHub URL
                    if len(match.groups()) == 2:
                        owner, repo = match.groups()
                        return f"https://github.com/{owner}/{repo}.git"
                    # Already normalized URL
                    elif url.endswith('.git'):
                        return url
                else:
                    # GitHub Enterprise URL
                    domain, owner, repo = match.groups()
                    return f"https://{domain}/{owner}/{repo}.git"
        
        # Not a recognized GitHub URL
        return None
    
    def _extract_repo_info(self, repo_path: Path) -> Dict[str, Any]:
        """Extract repository information"""
        repo = git.Repo(repo_path)
        
        # Get basic repository info
        try:
            repo_info = {
                "name": repo_path.name,
                "default_branch": repo.active_branch.name,
                "last_commit": {
                    "hash": repo.head.commit.hexsha,
                    "author": f"{repo.head.commit.author.name} <{repo.head.commit.author.email}>",
                    "date": repo.head.commit.committed_datetime.isoformat(),
                    "message": repo.head.commit.message.strip()
                },
                "commit_count": sum(1 for _ in repo.iter_commits()),
                "branches": [branch.name for branch in repo.branches],
                "readme": None
            }
            
            # Try to find README file
            readme_paths = ["README.md", "README.rst", "README.txt", "README"]
            for readme_path in readme_paths:
                full_path = repo_path / readme_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        repo_info["readme"] = f.read()
                    repo_info["readme_path"] = readme_path
                    break
                    
            return repo_info
            
        except Exception as e:
            logger.error(f"Error extracting repo info: {e}")
            return {"name": repo_path.name, "error": str(e)}
    
    def _find_documentation_files(
        self, 
        repo_path: Path, 
        doc_folders: List[str],
        file_extensions: List[str],
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """Find documentation files in the repository"""
        doc_files = []
        
        # Compile exclude patterns for efficient matching
        compiled_excludes = []
        if exclude_patterns:
            compiled_excludes = [re.compile(pattern) for pattern in exclude_patterns]
        
        # Special case for README.md in the root
        if "README.md" in doc_folders:
            readme_path = repo_path / "README.md"
            if readme_path.exists():
                doc_files.append(readme_path)
        
        # Find all documentation folders
        for folder_name in doc_folders:
            if folder_name.endswith(tuple(file_extensions)):
                # If it's a file pattern like README.md
                for path in repo_path.glob(f"**/{folder_name}"):
                    if path.is_file() and not any(pattern.search(str(path)) for pattern in compiled_excludes):
                        doc_files.append(path)
            else:
                # It's a directory
                for ext in file_extensions:
                    doc_dir = repo_path / folder_name
                    if doc_dir.exists() and doc_dir.is_dir():
                        for path in doc_dir.glob(f"**/*{ext}"):
                            if path.is_file() and not any(pattern.search(str(path)) for pattern in compiled_excludes):
                                doc_files.append(path)
        
        return sorted(doc_files)
    
    def _process_documentation_file(self, file_path: Path, repo_path: Path, repo_url: str) -> Dict[str, Any]:
        """Process a documentation file based on its extension"""
        try:
            # Get relative path for URL construction
            rel_path = file_path.relative_to(repo_path)
            # Convert to forward slashes for URL paths
            rel_path_str = self._normalize_path(str(rel_path))
            
            # Generate GitHub URL to the file
            file_url = f"{repo_url.replace('.git', '')}/blob/main/{rel_path_str}"
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Process based on file extension
            ext = file_path.suffix.lower()
            
            # Extract title from file content or use filename
            title = self._extract_doc_title(content, file_path.name)
            
            # Process content based on file type
            if ext == '.md':
                # Markdown files are already in markdown format
                processed_content = content
            elif ext == '.rst':
                # Convert RST to markdown (simplified version)
                processed_content = self._convert_rst_to_markdown(content)
            else:
                # For other formats, just use plain text
                processed_content = f"```\n{content}\n```"
            
            # Create result object
            result = {
                "title": title,
                "file_path": str(rel_path),
                "file_url": file_url,
                "content": processed_content,
                "content_type": ext[1:],  # Remove the dot
                "file_size": file_path.stat().st_size,
                "metadata": {
                    "repo_url": repo_url,
                    "file_extension": ext,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing documentation file {file_path}: {e}")
            return None
    
    def _extract_doc_title(self, content: str, default_filename: str) -> str:
        """Extract title from documentation content"""
        # Try to find a Markdown heading
        md_heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if md_heading_match:
            return md_heading_match.group(1).strip()
        
        # Try to find an RST heading
        rst_heading_match = re.search(r'^(.+)\n[=\-]+\n', content)
        if rst_heading_match:
            return rst_heading_match.group(1).strip()
        
        # Fallback to filename without extension
        return os.path.splitext(default_filename)[0]
    
    def _convert_rst_to_markdown(self, rst_content: str) -> str:
        """Simple conversion of RST to markdown format"""
        # This is a simplified conversion, for a complete conversion
        # you would use a library like docutils
        
        # Convert heading syntax
        md_content = rst_content
        
        # Replace === headings with # headings
        md_content = re.sub(r'(.+)\n={3,}\n', r'# \1\n', md_content)
        
        # Replace --- headings with ## headings
        md_content = re.sub(r'(.+)\n-{3,}\n', r'## \1\n', md_content)
        
        # Replace :code: and other roles
        md_content = re.sub(r':code:`([^`]+)`', r'`\1`', md_content)
        md_content = re.sub(r':py:func:`([^`]+)`', r'`\1()`', md_content)
        md_content = re.sub(r':ref:`([^<]+)<[^>]+>`', r'\1', md_content)
        
        # Replace hyperlinks
        md_content = re.sub(r'`([^<]+)<([^>]+)>`_', r'[\1](\2)', md_content)
        
        # Replace literal blocks
        md_content = re.sub(r'::\n\n', r'```\n', md_content)
        
        return md_content
    
    def _extract_code_examples(self, repo_path: Path, exclude_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract code examples from the repository"""
        code_files = []
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.rb', '.go', '.php', '.ipynb']
        
        # Compile exclude patterns for efficient matching
        compiled_excludes = []
        if exclude_patterns:
            compiled_excludes = [re.compile(pattern) for pattern in exclude_patterns]
            
        # Add common patterns to exclude
        common_excludes = [
            re.compile(r'node_modules'),
            re.compile(r'__pycache__'),
            re.compile(r'\.git'),
            re.compile(r'\.github'),
            re.compile(r'test_'),
            re.compile(r'_test\.'),
            re.compile(r'tests/'),
            re.compile(r'test/'),
            re.compile(r'\.venv'),
            re.compile(r'\.env')
        ]
        compiled_excludes.extend(common_excludes)
        
        # Find example code files
        for ext in code_extensions:
            for path in repo_path.glob(f"**/examples/**/*{ext}"):
                if not any(pattern.search(str(path)) for pattern in compiled_excludes):
                    code_files.append(path)
                    
            # Look for sample code in documentation directories
            for path in repo_path.glob(f"**/docs/**/*{ext}"):
                if not any(pattern.search(str(path)) for pattern in compiled_excludes):
                    code_files.append(path)
        
        # Limit to reasonable number
        code_files = code_files[:100]
        
        # Process each code file
        examples = []
        for file_path in code_files:
            try:
                # Get relative path for URL construction
                rel_path = file_path.relative_to(repo_path)
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Skip very large files
                if len(content) > 100000:  # ~100KB limit
                    continue
                
                # Extract docstring or file header comment if present
                description = self._extract_code_description(content, file_path.suffix)
                
                # Create result object
                examples.append({
                    "title": file_path.name,
                    "file_path": str(rel_path),
                    "content": content,
                    "language": file_path.suffix[1:],  # Remove the dot
                    "description": description
                })
                
            except Exception as e:
                logger.error(f"Error processing code example {file_path}: {e}")
        
        return examples
    
    def _extract_code_description(self, content: str, file_extension: str) -> Optional[str]:
        """Extract description from code file (docstring or header comment)"""
        # Extract Python docstring
        if file_extension == '.py':
            docstring_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
            if docstring_match:
                return docstring_match.group(1).strip()
                
        # Extract JS/TS block comment
        if file_extension in ['.js', '.ts', '.jsx', '.tsx']:
            comment_match = re.search(r'/\*\*(.+?)\*/', content, re.DOTALL)
            if comment_match:
                return comment_match.group(1).strip()
        
        # Extract any initial line comments
        first_lines = content.split('\n')[:10]
        comment_lines = []
        
        # Different comment markers for different languages
        comment_marker = '#' if file_extension in ['.py', '.rb'] else '//'
        
        for line in first_lines:
            line = line.strip()
            if line.startswith(comment_marker):
                comment_lines.append(line[len(comment_marker):].strip())
            elif not line and comment_lines:  # Keep empty lines within comments
                comment_lines.append('')
            elif comment_lines:  # Stop at first non-comment line after comments
                break
                
        if comment_lines:
            return '\n'.join(comment_lines)
            
        return None
        
    def _normalize_path(self, path: str) -> str:
        """Normalize file paths to use forward slashes for consistency across platforms"""
        return str(path).replace('\\', '/')
    
    async def shutdown(self):
        """Shutdown the scraper service, releasing resources"""
        logger.info("Shutting down scraper service")
        
        # Close browser
        await self.close_browser()
        
        # Cancel any active tasks
        for task in self._active_tasks:
            if not task.done() and not task.cancelled():
                task.cancel()
        
        # Wait for tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
            
        logger.info("Scraper service shutdown complete")