from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from app.api import filesystem, memory, git, scraper, documents, scheduler
from app.utils.config import get_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Store initialized services at module level for global access
app_services = {}

# API description for OpenAPI docs
API_DESCRIPTION = "A unified server providing filesystem, memory, git, web scraping, and document management tools for LLMs via OpenWebUI."

@asynccontextmanager
async def lifespan(app):
    """Lifespan handler for FastAPI to initialize and clean up services"""
    logger.info("Starting up server and initializing services")
    
    # Initialize services
    from app.core.scheduler_service import SchedulerService
    from app.core.scraper_service import ScraperService
    from app.api.scheduler import router as scheduler_router
    from app.api.scraper import scraper_service
    
    # Ensure data directories exist
    import os
    os.makedirs("./data/scraper", exist_ok=True)
    os.makedirs("./data/scraper/scheduled_jobs", exist_ok=True)
    os.makedirs("./data/scraper/scheduled_jobs/runs", exist_ok=True)
    
    # Create scheduler service instance
    scheduler_service = SchedulerService(
        scraper_service=scraper_service  # Use existing scraper service
    )
    
    # Store all services in module variable for global access
    app_services['scheduler'] = scheduler_service
    app_services['scraper'] = scraper_service
    
    # Make scheduler service available to the API router
    scheduler_router.scheduler_service = scheduler_service
    
    # Start the scheduler service
    scheduler_service.start()
    logger.info("All services initialized and started")
    
    yield
    
    # Shutdown all services in reverse initialization order
    logger.info("Shutting down server and cleaning up services")
    
    if 'scheduler' in app_services:
        app_services['scheduler'].shutdown()
        logger.info("Scheduler service shut down")
    
    # Shutdown scraper service to release browser resources
    if 'scraper' in app_services:
        # Ensure we use await for async shutdown
        await app_services['scraper'].shutdown()
        logger.info("Scraper service shut down")
        
    logger.info("All services shut down successfully")

app = FastAPI(
    title="OtherTales System Tools",
    version="1.0.0",
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# Configure CORS specifically for Open WebUI compatibility
origins = [
    # In production, remove the wildcard "*" and list only trusted domains
    # "*",  # Too permissive for production
    "https://ai.othertales.co",
    "https://legal.othertales.co",
    "https://mixture.othertales.co",
    "https://openwebui.com",
    # Add more specific domains as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Length"],
    max_age=600,  # Cache CORS preflight requests
)

# API Key Authentication dependency
async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    require_admin: bool = False
):
    # Skip API key verification if disabled
    if not config.api_key_required:
        return True
    
    if x_api_key is None:
        # Check for api_key in query params as fallback (less secure method)
        x_api_key = request.query_params.get("api_key")
        
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Get API keys from config
    default_api_key = config.api_keys.get("default", "")
    admin_api_key = config.api_keys.get("admin", "")
    
    # For admin-only operations, require admin key
    if require_admin:
        if x_api_key != admin_api_key:
            # Use 403 instead of 401 to indicate key was provided but lacks permissions
            raise HTTPException(
                status_code=403,
                detail="Admin API Key is required for this operation",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return True
    
    # For standard operations, allow either default or admin key
    if x_api_key != default_api_key and x_api_key != admin_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return True

# Admin-only dependency
async def admin_only(
    request: Request,
    x_api_key: Optional[str] = Header(None),
):
    return await verify_api_key(request, x_api_key, require_admin=True)

# Mount API routers with appropriate prefixes and tags
app.include_router(
    filesystem.router,
    prefix="/filesystem",
    tags=["Filesystem"],
    dependencies=[Depends(verify_api_key)] if config.api_key_required else []
)

app.include_router(
    memory.router,
    prefix="/memory",
    tags=["Memory"],
    dependencies=[Depends(verify_api_key)] if config.api_key_required else []
)

app.include_router(
    git.router,
    prefix="/git",
    tags=["Git"],
    dependencies=[Depends(verify_api_key)] if config.api_key_required else []
)

app.include_router(
    scraper.router,
    prefix="/scraper",
    tags=["Web Scraper"],
    dependencies=[Depends(verify_api_key)] if config.api_key_required else []
)

app.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)] if config.api_key_required else []
)

app.include_router(
    scheduler.router,
    prefix="/scheduler",
    tags=["Scheduler"],
    dependencies=[Depends(verify_api_key)] if config.api_key_required else []
)

# Create admin router for system management
from fastapi import APIRouter

admin_router = APIRouter()

@admin_router.get("/info")
async def admin_info():
    """Get system information for administration"""
    from os import path
    import sys
    import platform
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "app_version": app.version,
        "config": {
            # Provide only non-sensitive configuration
            "dev_mode": config.dev_mode,
            "admin_endpoints_enabled": config.admin_endpoints_enabled,
            "api_key_required": config.api_key_required,
            "file_cache_enabled": config.file_cache_enabled,
            "file_cache_max_age": config.file_cache_max_age,
            "vector_embedding_enabled": config.vector_embedding_enabled,
            "vector_model_name": config.vector_model_name,
            "search_provider": config.search_provider,
            "use_graph_db": config.use_graph_db,
        }
    }

@admin_router.post("/settings")
async def update_settings(settings: Dict[str, Any]):
    """Update server settings (admin only)"""
    # Only allow changing specific settings
    allowed_settings = [
        "api_key_required",
        "vector_embedding_enabled",
        "knowledge_graph_auto_link",
        "scraper_min_delay",
        "scraper_max_delay",
        "user_agent",
    ]
    
    updated = {}
    failed = {}
    
    try:
        for key, value in settings.items():
            if key in allowed_settings:
                try:
                    # Type checking for each setting
                    if key in ["api_key_required", "vector_embedding_enabled", "knowledge_graph_auto_link"]:
                        value = bool(value)
                    elif key in ["scraper_min_delay", "scraper_max_delay"]:
                        value = float(value)
                    
                    # Update setting
                    setattr(config, key, value)
                    updated[key] = value
                except (ValueError, TypeError) as e:
                    failed[key] = f"Invalid value type: {str(e)}"
            else:
                failed[key] = "Setting not allowed"
                
        return {
            "updated": updated,
            "failed": failed if failed else None
        }
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

# Include admin router if admin endpoints are enabled
if config.admin_endpoints_enabled:
    app.include_router(
        admin_router,
        prefix="/admin",
        tags=["Administration"],
        dependencies=[Depends(admin_only)]
    )

# Custom OpenAPI schema generator optimized for Open WebUI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="OtherTales Unified Tools Server",
        version=app.version,
        description="Document storage and retrieval system with Git versioning, knowledge graph, and web scraping capabilities.",
        routes=app.routes,
    )
    
    # Add tool metadata for Open WebUI
    openapi_schema["info"]["x-logo"] = {
        "url": "https://cdn-icons-png.flaticon.com/512/8728/8728086.png"
    }
    
    # Add toolkit info for better Open WebUI integration
    openapi_schema["info"]["x-openwebui-toolkit"] = {
        "category": "document-management",
        "capabilities": [
            "document-storage", 
            "web-scraping", 
            "git-versioning", 
            "memory",
            "knowledge-graph",
            "scheduled-tasks",
            "verification"
        ],
        "auth_required": config.api_key_required,
        "auth_method": "api_key" if config.api_key_required else None,
    }
    
    # Add security schemes if API key is required
    if config.api_key_required:
        openapi_schema["components"] = {
            "securitySchemes": {
                "APIKeyHeader": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "APIKeyQuery": {
                    "type": "apiKey",
                    "in": "query",
                    "name": "api_key"
                }
            }
        }
        
        openapi_schema["security"] = [
            {"APIKeyHeader": []},
            {"APIKeyQuery": []}
        ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Set custom OpenAPI schema generator
app.openapi = custom_openapi

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "OtherTales Unified Tools Server API",
        "services": ["filesystem", "memory", "git", "scraper", "documents", "scheduler"],
        "version": "1.0.0",
        "openapi_url": "/openapi.json",
        "auth_required": config.api_key_required,
    }

if __name__ == "__main__":
    import uvicorn
    
    # Log server startup information
    logger.info(f"Starting server on {config.server_host}:{config.server_port}")
    logger.info(f"API key authentication: {'enabled' if config.api_key_required else 'disabled'}")
    logger.info(f"Admin endpoints: {'enabled' if config.admin_endpoints_enabled else 'disabled'}")
    logger.info(f"Scheduler and verification: enabled")
    
    # Run production server
    uvicorn.run(
        "main:app", 
        host=config.server_host, 
        port=config.server_port, 
        reload=config.dev_mode,
        log_level="info"
    )