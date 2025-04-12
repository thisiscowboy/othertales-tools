from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from app.models.scraper import (
    ScrapeSingleUrlRequest,
    UrlList,
    ScrapeCrawlRequest,
    SearchAndScrapeRequest,
    SitemapScrapeRequest,
    GitHubRepoRequest,
    ScraperResponse
)
from app.core.scraper_service import ScraperService
from app.core.documents_service import DocumentsService
from app.models.documents import DocumentType
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
scraper_service = ScraperService()
documents_service = DocumentsService()


@router.post(
    "/url",
    response_model=ScraperResponse,
    summary="Scrape a single URL",
    description="Extract content from a web page and convert to Markdown",
)
async def scrape_url(request: ScrapeSingleUrlRequest = Body(...)):
    """
    Scrape a single URL and return structured data.
    Extracts content, converts to Markdown, and optionally stores as a document.
    """
    try:
        result = await scraper_service.scrape_url(
            request.url, request.wait_for_selector, request.wait_for_timeout
        )
        # If requested, store as document
        if request.store_as_document and result["success"]:
            doc = documents_service.create_document(
                title=result["title"],
                content=result["content"],
                document_type=DocumentType.WEBPAGE,
                metadata=result["metadata"],
                tags=request.document_tags,
                source_url=result["url"],
            )
            result["document_id"] = doc["id"]
        return result
    except Exception as e:
        logger.error(f"Error scraping URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@router.post(
    "/urls",
    response_model=List[ScraperResponse],
    summary="Scrape multiple URLs",
    description="Scrape multiple URLs in parallel",
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
                            source_url=result["url"],
                        )
                        results[i]["document_id"] = doc["id"]
                    except Exception as e:
                        results[i]["error"] = f"Document creation failed: {str(e)}"
        return results
    except Exception as e:
        logger.error(f"Error scraping URLs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@router.post(
    "/crawl",
    response_model=Dict[str, Any],
    summary="Crawl website",
    description="Crawl a website starting from a URL and store results in knowledge graph",
)
async def crawl_website(request: ScrapeCrawlRequest = Body(...)):
    """
    Crawl a website starting from a URL and store results in knowledge graph.
    Follows links up to a specified depth and processes each page.
    Optional verification pass ensures content stability.
    
    Special site_type support for:
    - 'legislation': Optimized for legislation.gov.uk (UK legislation)
    - 'hmrc': Optimized for gov.uk/hmrc (tax documentation)
    """
    try:
        # Get global config settings
        from app.utils.config import get_config
        config = get_config()
        
        # Start the crawl process
        logger.info(f"Starting crawl of {request.start_url} with max pages: {request.max_pages}, depth: {request.recursion_depth}")
        
        # Pass all the new parameters to the crawl_website method
        results = await scraper_service.crawl_website(
            start_url=request.start_url,
            max_pages=request.max_pages,
            recursion_depth=request.recursion_depth,
            allowed_domains=request.allowed_domains,
            verification_pass=request.verification_pass,
            site_type=request.site_type,
            follow_subdomains=request.follow_subdomains,
            content_types=request.content_types,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
        )
        
        response = {
            "pages_crawled": results.get("pages_crawled", 0),
            "start_url": request.start_url,
            "success_count": results.get("success_count", 0),
            "failed_count": results.get("failed_count", 0),
            "skipped_count": results.get("skipped_count", 0),
        }
        
        # Include verification results if available
        if "verification_results" in results:
            response["verification_results"] = results["verification_results"]
            response["verification_success_rate"] = results["verification_success_rate"]
            
        # If requested, create documents
        if request.create_documents:
            document_ids = []
            successes = 0
            failures = 0
            
            logger.info(f"Creating documents from crawled pages ({len(results.get('results', []))} pages)")
            
            # Choose appropriate document type based on site_type
            document_type = DocumentType.WEBPAGE
            if request.site_type == "legislation":
                document_type = DocumentType.LEGAL
            elif request.site_type == "hmrc":
                document_type = DocumentType.ACCOUNTANCY
            
            # Process crawled pages and create documents with knowledge graph integration
            for result in results.get("results", []):
                if result.get("success", False):
                    try:
                        # Skip if content is minimal
                        if len(result.get("content", "")) < 100:
                            logger.warning(f"Skipping document creation for URL {result['url']} - content too short")
                            failures += 1
                            continue
                            
                        # Prepare metadata with source information
                        metadata = result.get("metadata", {})
                        metadata.update({
                            "crawl_timestamp": result.get("scraped_at"),
                            "domain": result.get("domain", "unknown"),
                            "crawl_depth": result.get("depth", 0),
                        })
                        
                        # Auto-determine tags based on site type
                        tags = request.document_tags or []
                        if request.site_type == "legislation":
                            if "legislation" not in tags:
                                tags.append("legislation")
                            if "legal" not in tags:
                                tags.append("legal")
                            # Add specific legislation category based on URL pattern
                            url = result.get("url", "")
                            if "/ukpga/" in url:
                                tags.append("primary-legislation")
                            elif "/uksi/" in url:
                                tags.append("secondary-legislation")
                        elif request.site_type == "hmrc":
                            if "hmrc" not in tags:
                                tags.append("hmrc")
                            if "tax" not in tags:
                                tags.append("tax")
                        
                        # Create document with proper knowledge graph and embedding settings
                        doc = documents_service.create_document(
                            title=result["title"],
                            content=result["content"],
                            document_type=document_type,
                            metadata=metadata,
                            tags=tags,
                            source_url=result["url"],
                            # Use global settings for knowledge graph and embeddings
                            enable_vector_embedding=config.vector_embedding_enabled,
                            link_to_knowledge_graph=config.knowledge_graph_auto_link,
                        )
                        document_ids.append(doc["id"])
                        successes += 1
                    except Exception as doc_error:
                        logger.error(f"Error creating document for URL {result.get('url')}: {doc_error}")
                        failures += 1
                        
            response["documents_created"] = successes
            response["document_creation_failures"] = failures
            response["document_ids"] = document_ids
            
        return response
    except Exception as e:
        logger.error(f"Error crawling website: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crawling failed: {str(e)}")


@router.post(
    "/search",
    response_model=List[ScraperResponse],
    summary="Search and scrape",
    description="Search for content and scrape the results",
)
async def search_and_scrape(request: SearchAndScrapeRequest = Body(...)):
    """
    Search for content and scrape the results.
    Performs a web search and scrapes the top results.
    """
    try:
        results = await scraper_service.search_and_scrape(request.query, request.max_results)
        
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
                            source_url=result["url"],
                        )
                        results[i]["document_id"] = doc["id"]
                    except Exception:
                        pass
                        
        return results
    except Exception as e:
        logger.error(f"Error searching and scraping: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search and scrape failed: {str(e)}")


@router.post(
    "/sitemap",
    response_model=Dict[str, Any],
    summary="Scrape sitemap",
    description="Extract URLs from sitemap, scrape them, and store in the knowledge graph",
)
async def scrape_sitemap(request: SitemapScrapeRequest = Body(...)):
    """
    Extract URLs from a sitemap and scrape them.
    Processes XML sitemap files and scrapes the listed URLs.
    Optionally stores the results in the knowledge graph.
    """
    try:
        # Get global config settings
        from app.utils.config import get_config
        config = get_config()
        
        logger.info(f"Scraping sitemap from {request.sitemap_url} with max URLs: {request.max_urls}")
        result = await scraper_service.scrape_sitemap(request.sitemap_url, request.max_urls)
        
        # Handle document creation if requested
        if request.create_documents and result.get("urls_scraped", []):
            document_ids = []
            successes = 0
            failures = 0
            
            logger.info(f"Creating documents from sitemap URLs ({len(result.get('urls_scraped', []))} URLs)")
            
            # Create tags with sitemap source information
            tags = request.document_tags or []
            if "sitemap" not in tags:
                tags.append("sitemap")
            
            # Process each scraped URL and create documents
            for scraped_url in result["urls_scraped"]:
                if scraped_url.get("success", False):
                    try:
                        # Skip if content is minimal
                        if len(scraped_url.get("content", "")) < 100:
                            logger.warning(f"Skipping document creation for URL {scraped_url['url']} - content too short")
                            failures += 1
                            continue
                            
                        # Add sitemap source to metadata
                        metadata = scraped_url.get("metadata", {})
                        metadata.update({
                            "sitemap_source": request.sitemap_url,
                            "sitemap_scraped_at": scraped_url.get("scraped_at"),
                        })
                        
                        # Create document with knowledge graph and vector embedding
                        doc = documents_service.create_document(
                            title=scraped_url["title"],
                            content=scraped_url["content"],
                            document_type=DocumentType.WEBPAGE,
                            metadata=metadata,
                            tags=tags,
                            source_url=scraped_url["url"],
                            # Use global settings for knowledge graph and embeddings
                            enable_vector_embedding=config.vector_embedding_enabled,
                            link_to_knowledge_graph=config.knowledge_graph_auto_link,
                        )
                        document_ids.append(doc["id"])
                        successes += 1
                    except Exception as doc_error:
                        logger.error(f"Error creating document for URL {scraped_url.get('url')}: {doc_error}")
                        failures += 1
                        
            result["documents_created"] = successes
            result["document_creation_failures"] = failures
            result["document_ids"] = document_ids
            
        return result
    except Exception as e:
        logger.error(f"Error scraping sitemap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sitemap scraping failed: {str(e)}")


@router.post(
    "/github",
    response_model=Dict[str, Any],
    summary="Process GitHub repository",
    description="Extract documentation and code examples from a GitHub repository and store in the SDK Documentation knowledge graph",
)
async def process_github_repository(request: GitHubRepoRequest = Body(...)):
    """
    Extract documentation and code examples from a GitHub repository.
    
    Processes a GitHub repository to extract documentation files (Markdown, RST, etc.) 
    and code examples, optionally storing them in the SDK Documentation knowledge graph.
    
    Focuses on documentation folders like 'docs', 'examples', 'cookbook', etc.
    Identifies and processes code examples with their documentation.
    
    Useful for building knowledge bases for API/SDK documentation to support
    developer/coding assistant use cases.
    """
    try:
        # Get global config settings
        from app.utils.config import get_config
        config = get_config()
        
        logger.info(f"Processing GitHub repository: {request.repo_url}")
        
        # Call scraper service to process the GitHub repository
        result = await scraper_service.scrape_github_repository(
            repo_url=request.repo_url,
            auth_token=request.auth_token,
            doc_folders=request.doc_folders,
            file_extensions=request.file_extensions,
            exclude_patterns=request.exclude_patterns,
            extract_code_examples=request.extract_code_examples,
            max_files=request.max_files
        )
        
        # If the operation failed, return error
        if not result.get("success", False):
            return result
        
        # Handle document creation if requested
        if request.create_documents:
            document_ids = []
            doc_successes = 0
            code_successes = 0
            failures = 0
            
            # Create base tags with GitHub info
            tags = request.document_tags or []
            base_tags = tags.copy()
            if "github" not in base_tags:
                base_tags.append("github")
            if "sdk" not in base_tags:
                base_tags.append("sdk")
            if "documentation" not in base_tags:
                base_tags.append("documentation")
            
            # Determine document type from request
            if request.document_type == "API_REFERENCE":
                document_type = DocumentType.API_REFERENCE
            elif request.document_type == "SDK_GUIDE":
                document_type = DocumentType.SDK_GUIDE
            elif request.document_type == "CODE_EXAMPLE":
                document_type = DocumentType.CODE_EXAMPLE
            else:
                document_type = DocumentType.DOCUMENTATION
            
            # Process each documentation file
            logger.info(f"Creating documents from GitHub repo ({len(result.get('documents', []))} files)")
            for doc in result.get("documents", []):
                try:
                    # Skip if content is minimal
                    if not doc or len(doc.get("content", "")) < 50:
                        failures += 1
                        continue
                    
                    # Add file path information to tags
                    doc_tags = base_tags.copy()
                    file_path = doc.get("file_path", "")
                    # Add folder name as tag if it's a helpful categorization
                    parts = file_path.split("/")
                    if parts and parts[0] in ["docs", "guide", "reference", "examples"]:
                        doc_tags.append(parts[0])
                    
                    # Process content to ensure it's properly formatted
                    title = doc.get("title", "Untitled Document")
                    content = doc.get("content", "")
                    
                    # Create metadata with GitHub information
                    metadata = doc.get("metadata", {})
                    metadata.update({
                        "repo_url": request.repo_url,
                        "file_path": doc.get("file_path"),
                        "file_url": doc.get("file_url"),
                        "content_type": doc.get("content_type", "unknown")
                    })
                    
                    # Create document with knowledge graph and vector embedding
                    doc_obj = documents_service.create_document(
                        title=title,
                        content=content,
                        document_type=document_type,
                        metadata=metadata,
                        tags=doc_tags,
                        source_url=doc.get("file_url"),
                        # Use global settings for knowledge graph and embeddings
                        enable_vector_embedding=config.vector_embedding_enabled,
                        link_to_knowledge_graph=config.knowledge_graph_auto_link,
                    )
                    document_ids.append(doc_obj["id"])
                    doc_successes += 1
                except Exception as doc_error:
                    logger.error(f"Error creating document from GitHub file: {doc_error}")
                    failures += 1
            
            # Process code examples if available
            for example in result.get("code_examples", []):
                try:
                    if not example or len(example.get("content", "")) < 50:
                        continue
                    
                    # Add language tag and code example tag
                    code_tags = base_tags.copy()
                    language = example.get("language", "")
                    if language:
                        code_tags.append(language)
                    code_tags.append("code-example")
                    
                    # Format content with language markdown code block
                    title = example.get("title", "Untitled Example")
                    language_md = example.get("language", "")
                    description = example.get("description", "")
                    
                    # Create formatted content with description and code
                    content = f"# {title}\n\n"
                    if description:
                        content += f"{description}\n\n"
                    content += f"```{language_md}\n{example.get('content', '')}\n```"
                    
                    # Create metadata with GitHub information
                    metadata = {
                        "repo_url": request.repo_url,
                        "file_path": example.get("file_path"),
                        "language": example.get("language"),
                        "type": "code_example"
                    }
                    
                    # Create document with knowledge graph and vector embedding
                    doc_obj = documents_service.create_document(
                        title=title,
                        content=content,
                        document_type=DocumentType.CODE_EXAMPLE,
                        metadata=metadata,
                        tags=code_tags,
                        source_url=f"{request.repo_url.replace('.git', '')}/blob/main/{example.get('file_path', '')}",
                        # Use global settings for knowledge graph and embeddings
                        enable_vector_embedding=config.vector_embedding_enabled,
                        link_to_knowledge_graph=config.knowledge_graph_auto_link,
                    )
                    document_ids.append(doc_obj["id"])
                    code_successes += 1
                except Exception as code_error:
                    logger.error(f"Error creating document from code example: {code_error}")
                    failures += 1
            
            # Add document creation results to the response
            result["documents_created"] = doc_successes + code_successes
            result["documentation_documents"] = doc_successes
            result["code_example_documents"] = code_successes
            result["document_creation_failures"] = failures
            result["document_ids"] = document_ids
        
        return result
    except Exception as e:
        logger.error(f"Error processing GitHub repository: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GitHub repository processing failed: {str(e)}")