from fastapi import APIRouter, Body, HTTPException, Query, Path
from typing import Dict, Any, List, Optional

from app.models.scraper import (
    ScrapeSingleUrlRequest,
    UrlList,
    ScrapeCrawlRequest,
    SearchAndScrapeRequest,
    ScraperResponse
)
from app.core.scraper_service import ScraperService
from app.core.documents_service import DocumentsService
from app.models.documents import DocumentType

router = APIRouter(
    responses={
        400: {"description": "Bad request"},
        500: {"description": "Scraping failed"}
    }
)

scraper_service = ScraperService()
documents_service = DocumentsService()

@router.post(
    "/url",
    response_model=ScraperResponse,
    summary="Scrape a single URL",
    description="Extract content from a web page and convert to Markdown"
)
async def scrape_url(request: ScrapeSingleUrlRequest = Body(...)):
    """
    Scrape a single URL and return structured data.
    
    Extracts content, converts to Markdown, and optionally stores as a document.
    """
    try:
        result = await scraper_service.scrape_url(
            request.url,
            request.wait_for_selector,
            request.wait_for_timeout
        )
        
        # If requested, store as document
        if request.store_as_document and result["success"]:
            doc = documents_service.create_document(
                title=result["title"],
                content=result["content"],
                document_type=DocumentType.WEBPAGE,
                metadata=result["metadata"],
                tags=request.document_tags,
                source_url=result["url"]
            )
            result["document_id"] = doc["id"]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@router.post(
    "/urls",
    response_model=List[ScraperResponse],
    summary="Scrape multiple URLs",
    description="Scrape multiple URLs in parallel"
)
async def scrape_multiple_urls(request: UrlList = Body(...)):
    """
    Scrape multiple URLs in parallel.
    
    Processes a list of URLs and returns the scraped content for each.
    """
    try:
        results = await scraper_service.scrape_urls(request.urls)
        
        # If requested, store results as documents
        if request.store_as_documents:
            for i, result in enumerate(results):
                if result["success"]:
                    try:
                        doc = documents_service.create_document(
                            title=result["title"],
                            content=result["content"],
                            document_type=DocumentType.WEBPAGE,
                            metadata=result["metadata"],
                            tags=request.document_tags,
                            source_url=result["url"]
                        )
                        results[i]["document_id"] = doc["id"]
                    except Exception as e:
                        results[i]["error"] = f"Document creation failed: {str(e)}"
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@router.post(
    "/crawl",
    response_model=Dict[str, Any],
    summary="Crawl website",
    description="Crawl a website starting from a URL"
)
async def crawl_website(request: ScrapeCrawlRequest = Body(...)):
    """
    Crawl a website starting from a URL.
    
    Follows links up to a specified depth and processes each page.
    Optional verification pass ensures content stability.
    """
    try:
        results = await scraper_service.crawl_website(
            request.start_url,
            request.max_pages,
            request.recursion_depth,
            request.allowed_domains,
            request.verification_pass  # Pass the verification_pass parameter
        )
        
        response = {
            "pages_crawled": results.get("pages_crawled", 0),
            "start_url": request.start_url,
            "success_count": results.get("success_count", 0),
            "failed_count": results.get("failed_count", 0)
        }
        
        # Include verification results if available
        if "verification_results" in results:
            response["verification_results"] = results["verification_results"]
            response["verification_success_rate"] = results["verification_success_rate"]
        
        # If requested, create documents
        if request.create_documents:
            document_ids = []
            for result in results.get("results", []):
                if result.get("success", False):
                    try:
                        doc = documents_service.create_document(
                            title=result["title"],
                            content=result["content"],
                            document_type=DocumentType.WEBPAGE,
                            metadata=result["metadata"],
                            tags=request.document_tags,
                            source_url=result["url"]
                        )
                        document_ids.append(doc["id"])
                    except Exception:
                        pass
            
            response["documents_created"] = len(document_ids)
            response["document_ids"] = document_ids
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawling failed: {str(e)}")

@router.post(
    "/search",
    response_model=List[ScraperResponse],
    summary="Search and scrape",
    description="Search for content and scrape the results"
)
async def search_and_scrape(request: SearchAndScrapeRequest = Body(...)):
    """
    Search for content and scrape the results.
    
    Performs a web search and scrapes the top results.
    """
    try:
        results = await scraper_service.search_and_scrape(
            request.query,
            request.max_results
        )
        
        # If requested, create documents
        if request.create_documents:
            for i, result in enumerate(results):
                if result.get("success", False):
                    try:
                        doc = documents_service.create_document(
                            title=result["title"],
                            content=result["content"],
                            document_type=DocumentType.WEBPAGE,
                            metadata=result["metadata"],
                            tags=request.document_tags,
                            source_url=result["url"]
                        )
                        results[i]["document_id"] = doc["id"]
                    except Exception:
                        pass
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search and scrape failed: {str(e)}")

@router.post(
    "/screenshot",
    response_model=Dict[str, Any],
    summary="Capture screenshot",
    description="Capture screenshot of a webpage"
)
async def capture_screenshot(
    url: str = Body(..., embed=True),
    full_page: bool = Body(True, embed=True)
):
    """
    Capture a screenshot of a URL.
    
    Returns the path to the saved screenshot file.
    """
    try:
        result = await scraper_service.capture_screenshot(url, full_page)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot failed: {str(e)}")
