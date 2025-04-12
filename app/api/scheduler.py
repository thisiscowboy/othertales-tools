from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body, Depends, Query
from app.core.scheduler_service import SchedulerService
from pydantic import BaseModel, Field
import logging
import asyncio
import json

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Scheduler service will be set from main.py
scheduler_service = None

# Define request/response models
class WebScrapeScheduleRequest(BaseModel):
    """Request to schedule a web scraping job"""
    job_id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Job name")
    start_url: str = Field(..., description="Starting URL for crawl")
    schedule_type: str = Field("interval", description="Schedule type: 'interval' or 'cron'")
    interval_seconds: Optional[int] = Field(None, description="Interval in seconds (for interval scheduling)")
    cron_expression: Optional[str] = Field(None, description="Cron expression (for cron scheduling)")
    site_type: Optional[str] = Field(None, description="Site type (legislation, hmrc, etc.)")
    max_pages: int = Field(5000, description="Maximum pages to crawl")
    verification_enabled: bool = Field(True, description="Whether to verify scraped content")
    allowed_domains: Optional[List[str]] = Field(None, description="Domains to restrict crawling to")
    include_patterns: Optional[List[str]] = Field(None, description="URL patterns to include")
    exclude_patterns: Optional[List[str]] = Field(None, description="URL patterns to exclude")
    document_tags: Optional[List[str]] = Field(None, description="Tags for documents")

class GithubScrapeScheduleRequest(BaseModel):
    """Request to schedule a GitHub repository scraping job"""
    job_id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Job name")
    repo_url: str = Field(..., description="GitHub repository URL")
    schedule_type: str = Field("interval", description="Schedule type: 'interval' or 'cron'")
    interval_seconds: Optional[int] = Field(None, description="Interval in seconds (for interval scheduling)")
    cron_expression: Optional[str] = Field(None, description="Cron expression (for cron scheduling)")
    auth_token: Optional[str] = Field(None, description="GitHub authentication token")
    doc_folders: Optional[List[str]] = Field(None, description="Documentation folders to focus on")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to include")
    max_files: int = Field(500, description="Maximum files to process")
    document_tags: Optional[List[str]] = Field(None, description="Tags for documents")

class VerificationScheduleRequest(BaseModel):
    """Request to schedule a document verification job"""
    job_id: str = Field(..., description="Unique job identifier")
    name: str = Field(..., description="Job name")
    document_tags: List[str] = Field(..., description="Tags to filter documents")
    document_type: Optional[str] = Field(None, description="Type of documents to verify")
    schedule_type: str = Field("interval", description="Schedule type: 'interval' or 'cron'")
    interval_seconds: Optional[int] = Field(None, description="Interval in seconds (for interval scheduling)")
    cron_expression: Optional[str] = Field(None, description="Cron expression (for cron scheduling)")
    max_documents: int = Field(1000, description="Maximum documents to verify")

class JobResponse(BaseModel):
    """Response for job operations"""
    success: bool = Field(..., description="Whether the operation was successful")
    job_id: str = Field(..., description="Job identifier")
    message: str = Field(..., description="Response message")

# API endpoints
@router.get(
    "/jobs",
    response_model=List[Dict[str, Any]],
    summary="Get scheduled jobs",
    description="Get list of all scheduled jobs"
)
async def get_scheduled_jobs():
    """Get all scheduled jobs"""
    try:
        jobs = scheduler_service.get_jobs()
        return jobs
    except Exception as e:
        logger.error(f"Error getting scheduled jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled jobs: {str(e)}")

@router.post(
    "/web-scrape",
    response_model=JobResponse,
    summary="Schedule web scraping job",
    description="Schedule a website scraping job to run periodically"
)
async def schedule_web_scrape(request: WebScrapeScheduleRequest = Body(...)):
    """Schedule a web scraping job"""
    try:
        result = scheduler_service.add_web_scrape_job(
            job_id=request.job_id,
            name=request.name,
            start_url=request.start_url,
            schedule_type=request.schedule_type,
            interval_seconds=request.interval_seconds,
            cron_expression=request.cron_expression,
            site_type=request.site_type,
            max_pages=request.max_pages,
            verification_enabled=request.verification_enabled,
            allowed_domains=request.allowed_domains,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
            document_tags=request.document_tags
        )
        return result
    except Exception as e:
        logger.error(f"Error scheduling web scrape job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")

@router.post(
    "/github-scrape",
    response_model=JobResponse,
    summary="Schedule GitHub repository scraping job",
    description="Schedule a GitHub repository scraping job to run periodically"
)
async def schedule_github_scrape(request: GithubScrapeScheduleRequest = Body(...)):
    """Schedule a GitHub repository scraping job"""
    try:
        result = scheduler_service.add_github_scrape_job(
            job_id=request.job_id,
            name=request.name,
            repo_url=request.repo_url,
            schedule_type=request.schedule_type,
            interval_seconds=request.interval_seconds,
            cron_expression=request.cron_expression,
            auth_token=request.auth_token,
            doc_folders=request.doc_folders,
            file_extensions=request.file_extensions,
            max_files=request.max_files,
            document_tags=request.document_tags
        )
        return result
    except Exception as e:
        logger.error(f"Error scheduling GitHub scrape job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")

@router.post(
    "/verification",
    response_model=JobResponse,
    summary="Schedule document verification job",
    description="Schedule a document verification job to run periodically"
)
async def schedule_verification(request: VerificationScheduleRequest = Body(...)):
    """Schedule a document verification job"""
    try:
        result = scheduler_service.add_verification_job(
            job_id=request.job_id,
            name=request.name,
            document_tags=request.document_tags,
            document_type=request.document_type,
            schedule_type=request.schedule_type,
            interval_seconds=request.interval_seconds,
            cron_expression=request.cron_expression,
            max_documents=request.max_documents
        )
        return result
    except Exception as e:
        logger.error(f"Error scheduling verification job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")

@router.delete(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Remove scheduled job",
    description="Remove a scheduled job"
)
async def remove_scheduled_job(job_id: str):
    """Remove a scheduled job"""
    try:
        result = scheduler_service.remove_job(job_id)
        return result
    except Exception as e:
        logger.error(f"Error removing scheduled job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove job: {str(e)}")

@router.post(
    "/jobs/{job_id}/run-now",
    response_model=JobResponse,
    summary="Run job immediately",
    description="Trigger a scheduled job to run immediately"
)
async def run_job_immediately(job_id: str):
    """Run a scheduled job immediately"""
    try:
        # Get job details
        jobs = scheduler_service.get_jobs()
        job = next((j for j in jobs if j["id"] == job_id), None)
        
        if not job:
            return JobResponse(
                success=False,
                job_id=job_id,
                message=f"Job {job_id} not found"
            )
            
        # Run the job based on its type
        job_type = job.get("job_type")
        
        if job_type == "web_scrape":
            # Run in the background without awaiting completion
            asyncio.create_task(scheduler_service._run_web_scrape_job(job))
            message = f"Web scrape job {job_id} triggered"
        elif job_type == "github_scrape":
            asyncio.create_task(scheduler_service._run_github_scrape_job(job))
            message = f"GitHub scrape job {job_id} triggered"
        elif job_type == "verification":
            asyncio.create_task(scheduler_service._run_verification_job(job))
            message = f"Verification job {job_id} triggered"
        else:
            return JobResponse(
                success=False,
                job_id=job_id,
                message=f"Unknown job type: {job_type}"
            )
            
        return JobResponse(
            success=True,
            job_id=job_id,
            message=message
        )
            
    except Exception as e:
        logger.error(f"Error running job immediately: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run job: {str(e)}")

@router.get(
    "/jobs/{job_id}/history",
    response_model=List[Dict[str, Any]],
    summary="Get job run history",
    description="Get history of job runs"
)
async def get_job_history(job_id: str, limit: int = Query(10, description="Maximum number of history entries")):
    """Get job run history"""
    try:
        # Get job history from files
        from pathlib import Path
        
        history = []
        runs_dir = scheduler_service.job_runs_dir
        
        # Find all history files for this job
        if runs_dir.exists():
            files = list(runs_dir.glob(f"{job_id}_*.json"))
            # Sort by timestamp (newest first)
            files.sort(reverse=True)
            
            # Read history files
            for file_path in files[:limit]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        run_data = json.loads(f.read())
                        history.append(run_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in job history file {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error reading job history file {file_path}: {e}")
        
        return history
    except Exception as e:
        logger.error(f"Error getting job history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job history: {str(e)}")