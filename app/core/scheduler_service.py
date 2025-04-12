import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path

# Import APScheduler components
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.base import JobLookupError

from app.utils.config import get_config
from app.core.scraper_service import ScraperService
from app.core.documents_service import DocumentsService
from app.models.documents import DocumentType

logger = logging.getLogger(__name__)

class SchedulerService:
    """Service for scheduling and managing periodic tasks"""

    def __init__(self, scraper_service: Optional[ScraperService] = None, 
                 documents_service: Optional[DocumentsService] = None):
        """Initialize the scheduler service"""
        self.config = get_config()
        self.scheduler = AsyncIOScheduler()
        self.scraper_service = scraper_service or ScraperService()
        self.documents_service = documents_service or DocumentsService()

        # Create directory for job data
        self.jobs_dir = Path(self.config.scraper_data_path) / "scheduled_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Job storage
        self.jobs_file = self.jobs_dir / "scheduled_jobs.json"
        self.job_runs_dir = self.jobs_dir / "runs"
        self.job_runs_dir.mkdir(exist_ok=True)

        # Load existing jobs
        self._load_jobs()

    def _load_jobs(self):
        """Load scheduled jobs from storage"""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    jobs_data = json.load(f)
                    
                # Schedule each job
                for job_id, job_data in jobs_data.items():
                    self._schedule_job_from_data(job_id, job_data)
                    
                logger.info(f"Loaded {len(jobs_data)} scheduled jobs")
            except Exception as e:
                logger.error(f"Error loading scheduled jobs: {e}")
                # Initialize with empty jobs if file is corrupted
                self._save_jobs({})
        else:
            # Initialize with empty jobs file
            self._save_jobs({})

    def _save_jobs(self, jobs_data: Dict[str, Any]):
        """Save scheduled jobs to storage"""
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(self.jobs_file), exist_ok=True)
            
            # Create a temp file for atomic write
            temp_file = f"{self.jobs_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(jobs_data, f, indent=2)
                
            # Ensure the file is written to disk
            f.flush()
            os.fsync(f.fileno())
            
            # Rename for atomic replace (works on both Windows and Unix)
            if os.path.exists(self.jobs_file):
                os.replace(temp_file, self.jobs_file)
            else:
                os.rename(temp_file, self.jobs_file)
                
        except Exception as e:
            logger.error(f"Error saving scheduled jobs: {e}", exc_info=True)

    def _get_current_jobs(self) -> Dict[str, Any]:
        """Get all current job configurations"""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading jobs file: {e}")
                return {}
        return {}

    def _schedule_job_from_data(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Schedule a job from saved data"""
        try:
            job_type = job_data.get("job_type")
            
            if job_type == "web_scrape":
                # Create the appropriate trigger
                if job_data.get("schedule_type") == "cron":
                    trigger = CronTrigger.from_crontab(job_data.get("cron_expression", "0 0 * * *"))
                else:
                    # Default to interval scheduling
                    interval_seconds = job_data.get("interval_seconds", 86400)  # Default to daily
                    trigger = IntervalTrigger(seconds=interval_seconds)
                
                # Schedule the job
                self.scheduler.add_job(
                    self._run_web_scrape_job,
                    trigger=trigger,
                    args=[job_data],
                    id=job_id,
                    replace_existing=True
                )
                return True
                
            elif job_type == "github_scrape":
                # Create the appropriate trigger
                if job_data.get("schedule_type") == "cron":
                    trigger = CronTrigger.from_crontab(job_data.get("cron_expression", "0 0 * * *"))
                else:
                    # Default to interval scheduling
                    interval_seconds = job_data.get("interval_seconds", 604800)  # Default to weekly
                    trigger = IntervalTrigger(seconds=interval_seconds)
                
                # Schedule the job
                self.scheduler.add_job(
                    self._run_github_scrape_job,
                    trigger=trigger,
                    args=[job_data],
                    id=job_id,
                    replace_existing=True
                )
                return True
                
            elif job_type == "verification":
                # Create the appropriate trigger
                if job_data.get("schedule_type") == "cron":
                    trigger = CronTrigger.from_crontab(job_data.get("cron_expression", "0 0 * * 0"))
                else:
                    # Default to interval scheduling
                    interval_seconds = job_data.get("interval_seconds", 604800)  # Default to weekly
                    trigger = IntervalTrigger(seconds=interval_seconds)
                
                # Schedule the job
                self.scheduler.add_job(
                    self._run_verification_job,
                    trigger=trigger,
                    args=[job_data],
                    id=job_id,
                    replace_existing=True
                )
                return True
                
            else:
                logger.error(f"Unknown job type: {job_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error scheduling job {job_id}: {e}")
            return False

    def start(self):
        """Start the scheduler"""
        try:
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info("Scheduler started successfully")
                
                # Report number of loaded jobs
                jobs_count = len(self.scheduler.get_jobs())
                if jobs_count > 0:
                    logger.info(f"Loaded {jobs_count} scheduled jobs")
            else:
                logger.warning("Scheduler already running - start request ignored")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}", exc_info=True)
            raise

    def shutdown(self):
        """Shutdown the scheduler"""
        try:
            if self.scheduler.running:
                # Save any pending job data
                jobs_data = self._get_current_jobs()
                self._save_jobs(jobs_data)
                
                # Shutdown scheduler
                self.scheduler.shutdown()
                logger.info("Scheduler shutdown successfully")
            else:
                logger.warning("Scheduler not running - shutdown request ignored")
        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {e}", exc_info=True)
            # Make a best effort to shutdown even after error
            try:
                if self.scheduler.running:
                    self.scheduler.shutdown(wait=False)
            except:
                pass

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            job_info = {
                "id": job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            
            # Get additional metadata from stored configuration
            jobs_data = self._get_current_jobs()
            if job.id in jobs_data:
                job_info.update(jobs_data[job.id])
            
            jobs.append(job_info)
            
        return jobs

    def add_web_scrape_job(self, 
                          job_id: str,
                          name: str,
                          start_url: str,
                          schedule_type: str = "interval",
                          interval_seconds: Optional[int] = None,
                          cron_expression: Optional[str] = None,
                          site_type: Optional[str] = None,
                          max_pages: int = 5000,
                          verification_enabled: bool = True,
                          allowed_domains: Optional[List[str]] = None,
                          include_patterns: Optional[List[str]] = None,
                          exclude_patterns: Optional[List[str]] = None,
                          document_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add a new web scraping job
        
        Args:
            job_id: Unique identifier for the job
            name: Descriptive name for the job
            start_url: Starting URL for crawling
            schedule_type: "interval" or "cron"
            interval_seconds: Seconds between runs (for interval scheduling)
            cron_expression: Cron expression (for cron scheduling)
            site_type: Site type identifier (legislation, hmrc, etc.)
            max_pages: Maximum pages to scrape
            verification_enabled: Whether to perform verification pass
            allowed_domains: Domains to restrict crawling to
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude
            document_tags: Tags for created documents
            
        Returns:
            Dict with job information
        """
        # Validate scheduling parameters
        if schedule_type == "interval" and not interval_seconds:
            interval_seconds = 86400  # Default to daily
        elif schedule_type == "cron" and not cron_expression:
            cron_expression = "0 0 * * *"  # Default to midnight daily
            
        # Create job data
        job_data = {
            "job_id": job_id,
            "name": name,
            "job_type": "web_scrape",
            "start_url": start_url,
            "schedule_type": schedule_type,
            "interval_seconds": interval_seconds,
            "cron_expression": cron_expression,
            "site_type": site_type,
            "max_pages": max_pages,
            "verification_enabled": verification_enabled,
            "allowed_domains": allowed_domains,
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "document_tags": document_tags,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "last_status": None
        }
        
        # Schedule the job
        success = self._schedule_job_from_data(job_id, job_data)
        
        if success:
            # Update jobs data
            jobs_data = self._get_current_jobs()
            jobs_data[job_id] = job_data
            self._save_jobs(jobs_data)
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Scheduled web scrape job: {name}"
            }
        else:
            return {
                "success": False,
                "job_id": job_id,
                "message": "Failed to schedule job"
            }

    def add_github_scrape_job(self,
                             job_id: str,
                             name: str,
                             repo_url: str,
                             schedule_type: str = "interval",
                             interval_seconds: Optional[int] = None,
                             cron_expression: Optional[str] = None,
                             auth_token: Optional[str] = None,
                             doc_folders: Optional[List[str]] = None,
                             file_extensions: Optional[List[str]] = None,
                             max_files: int = 500,
                             document_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add a new GitHub repository scraping job
        
        Args:
            job_id: Unique identifier for the job
            name: Descriptive name for the job
            repo_url: GitHub repository URL
            schedule_type: "interval" or "cron"
            interval_seconds: Seconds between runs (for interval scheduling)
            cron_expression: Cron expression (for cron scheduling)
            auth_token: GitHub authentication token
            doc_folders: Folders to focus on
            file_extensions: File extensions to include
            max_files: Maximum files to process
            document_tags: Tags for created documents
            
        Returns:
            Dict with job information
        """
        # Validate scheduling parameters
        if schedule_type == "interval" and not interval_seconds:
            interval_seconds = 604800  # Default to weekly
        elif schedule_type == "cron" and not cron_expression:
            cron_expression = "0 0 * * 0"  # Default to midnight Sundays
            
        # Use defaults from config if not specified
        if not doc_folders:
            doc_folders = self.config.github_doc_folders
            
        if not file_extensions:
            file_extensions = [".md", ".rst", ".txt", ".ipynb", ".py", ".js", ".ts", ".jsx", ".tsx"]
        
        # Create job data
        job_data = {
            "job_id": job_id,
            "name": name,
            "job_type": "github_scrape",
            "repo_url": repo_url,
            "schedule_type": schedule_type,
            "interval_seconds": interval_seconds,
            "cron_expression": cron_expression,
            "auth_token": auth_token,
            "doc_folders": doc_folders,
            "file_extensions": file_extensions,
            "max_files": max_files,
            "document_tags": document_tags,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "last_status": None
        }
        
        # Schedule the job
        success = self._schedule_job_from_data(job_id, job_data)
        
        if success:
            # Update jobs data
            jobs_data = self._get_current_jobs()
            jobs_data[job_id] = job_data
            self._save_jobs(jobs_data)
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Scheduled GitHub scrape job: {name}"
            }
        else:
            return {
                "success": False,
                "job_id": job_id,
                "message": "Failed to schedule job"
            }

    def add_verification_job(self,
                           job_id: str,
                           name: str,
                           document_tags: List[str],
                           document_type: Optional[str] = None,
                           schedule_type: str = "interval",
                           interval_seconds: Optional[int] = None,
                           cron_expression: Optional[str] = None,
                           max_documents: int = 1000) -> Dict[str, Any]:
        """
        Add a new document verification job
        
        Args:
            job_id: Unique identifier for the job
            name: Descriptive name for the job
            document_tags: Tags to filter documents
            document_type: Type of documents to verify
            schedule_type: "interval" or "cron"
            interval_seconds: Seconds between runs (for interval scheduling)
            cron_expression: Cron expression (for cron scheduling)
            max_documents: Maximum documents to verify
            
        Returns:
            Dict with job information
        """
        # Validate scheduling parameters
        if schedule_type == "interval" and not interval_seconds:
            interval_seconds = 604800  # Default to weekly
        elif schedule_type == "cron" and not cron_expression:
            cron_expression = "0 0 * * 0"  # Default to midnight Sundays
            
        # Create job data
        job_data = {
            "job_id": job_id,
            "name": name,
            "job_type": "verification",
            "document_tags": document_tags,
            "document_type": document_type,
            "schedule_type": schedule_type,
            "interval_seconds": interval_seconds,
            "cron_expression": cron_expression,
            "max_documents": max_documents,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "last_status": None
        }
        
        # Schedule the job
        success = self._schedule_job_from_data(job_id, job_data)
        
        if success:
            # Update jobs data
            jobs_data = self._get_current_jobs()
            jobs_data[job_id] = job_data
            self._save_jobs(jobs_data)
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Scheduled verification job: {name}"
            }
        else:
            return {
                "success": False,
                "job_id": job_id,
                "message": "Failed to schedule job"
            }

    def remove_job(self, job_id: str) -> Dict[str, Any]:
        """Remove a scheduled job"""
        try:
            # Remove from scheduler
            self.scheduler.remove_job(job_id)
            
            # Remove from persistent storage
            jobs_data = self._get_current_jobs()
            if job_id in jobs_data:
                del jobs_data[job_id]
                self._save_jobs(jobs_data)
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Removed job {job_id}"
            }
        except JobLookupError:
            return {
                "success": False,
                "job_id": job_id,
                "message": f"Job {job_id} not found"
            }
        except Exception as e:
            logger.error(f"Error removing job {job_id}: {e}")
            return {
                "success": False,
                "job_id": job_id,
                "message": f"Error removing job: {str(e)}"
            }

    def update_job_status(self, job_id: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Update job status after a run"""
        try:
            jobs_data = self._get_current_jobs()
            if job_id in jobs_data:
                jobs_data[job_id]["last_run"] = datetime.now().isoformat()
                jobs_data[job_id]["last_status"] = status
                if details:
                    jobs_data[job_id]["last_run_details"] = details
                self._save_jobs(jobs_data)
                
                # Save detailed run data to separate file
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_file = self.job_runs_dir / f"{job_id}_{run_id}.json"
                run_data = {
                    "job_id": job_id,
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "details": details or {}
                }
                
                with open(run_file, 'w', encoding='utf-8') as f:
                    json.dump(run_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating job status for {job_id}: {e}")

    # Job execution methods
    async def _run_web_scrape_job(self, job_data: Dict[str, Any]):
        """Run a web scraping job"""
        job_id = job_data.get("job_id")
        start_url = job_data.get("start_url")
        site_type = job_data.get("site_type")
        
        logger.info(f"Running scheduled web scrape job {job_id} for {start_url}")
        
        try:
            # Prepare scraping parameters
            params = {
                "start_url": start_url,
                "max_pages": job_data.get("max_pages", 5000),
                "allowed_domains": job_data.get("allowed_domains"),
                "verification_pass": job_data.get("verification_enabled", True),
                "site_type": site_type,
                "include_patterns": job_data.get("include_patterns"),
                "exclude_patterns": job_data.get("exclude_patterns")
            }
            
            # Run the scraping process
            result = await self.scraper_service.crawl_website(**params)
            
            # Process results
            success_count = result.get("success_count", 0)
            failed_count = result.get("failed_count", 0)
            skipped_count = result.get("skipped_count", 0)
            total_pages = result.get("pages_crawled", 0)
            
            # Calculate success rate
            success_rate = 0
            if total_pages > 0:
                success_rate = (success_count / total_pages) * 100
                
            # Check if additional verification needed
            verification_results = result.get("verification_results", [])
            verification_failed = []
            
            if verification_results:
                for vr in verification_results:
                    if not vr.get("verified", False):
                        verification_failed.append(vr.get("url"))
            
            # Process documents if scraping was successful
            document_ids = []
            document_type = DocumentType.WEBPAGE
            
            # Determine appropriate document type based on site_type
            if site_type == "legislation":
                document_type = DocumentType.LEGAL
            elif site_type == "hmrc":
                document_type = DocumentType.ACCOUNTANCY
                
            # Apply tags
            tags = job_data.get("document_tags") or []
            for result_item in result.get("results", []):
                if result_item.get("success", False):
                    try:
                        # Skip if content is minimal
                        content = result_item.get("content", "")
                        if len(content) < 100:
                            continue
                            
                        # Add metadata
                        metadata = result_item.get("metadata", {})
                        metadata.update({
                            "scheduled_job_id": job_id,
                            "scheduled_job_name": job_data.get("name"),
                            "crawl_timestamp": result_item.get("scraped_at")
                        })
                        
                        # Create document
                        doc = self.documents_service.create_document(
                            title=result_item["title"],
                            content=content,
                            document_type=document_type,
                            metadata=metadata,
                            tags=tags,
                            source_url=result_item["url"],
                            enable_vector_embedding=True,
                            link_to_knowledge_graph=True
                        )
                        document_ids.append(doc["id"])
                    except KeyError as e:
                        logger.error(f"Missing required field {e} for URL {result_item.get('url')}")
                    except ValueError as e:
                        logger.error(f"Invalid data for document creation: {e}")
                    except Exception as e:
                        logger.error(f"Error creating document for URL {result_item.get('url')}: {e}", exc_info=True)
            
            # Update job status
            status = "success" if success_rate >= 85 else "partial" if success_rate >= 50 else "failed"
            
            details = {
                "start_url": start_url,
                "site_type": site_type,
                "pages_crawled": total_pages,
                "success_count": success_count,
                "failed_count": failed_count,
                "skipped_count": skipped_count,
                "success_rate": success_rate,
                "verification_failed_count": len(verification_failed),
                "verification_failed_urls": verification_failed[:50],  # Limit for storage
                "documents_created": len(document_ids),
                "document_ids": document_ids[:100]  # Limit for storage
            }
            
            self.update_job_status(job_id, status, details)
            
            # If verification found failed pages, consider a follow-up job
            if verification_failed and job_data.get("verification_enabled", True):
                # Create a follow-up attempt for failed pages
                await self._process_verification_failures(job_id, job_data, verification_failed)
                
            logger.info(f"Completed scheduled web scrape job {job_id}: {status} ({success_rate:.2f}% success rate)")
            
        except Exception as e:
            logger.error(f"Error in scheduled web scrape job {job_id}: {e}", exc_info=True)
            self.update_job_status(job_id, "error", {"error": str(e)})

    async def _run_github_scrape_job(self, job_data: Dict[str, Any]):
        """Run a GitHub repository scraping job"""
        job_id = job_data.get("job_id")
        repo_url = job_data.get("repo_url")
        
        logger.info(f"Running scheduled GitHub scrape job {job_id} for {repo_url}")
        
        try:
            # Prepare scraping parameters
            params = {
                "repo_url": repo_url,
                "auth_token": job_data.get("auth_token"),
                "doc_folders": job_data.get("doc_folders"),
                "file_extensions": job_data.get("file_extensions"),
                "max_files": job_data.get("max_files", 500)
            }
            
            # Run the scraping process
            result = await self.scraper_service.scrape_github_repository(**params)
            
            # Process results
            success = result.get("success", False)
            files_found = result.get("files_found", 0)
            files_processed = result.get("files_processed", 0)
            
            # Process documents if scraping was successful
            document_ids = []
            doc_successes = 0
            code_successes = 0
            failures = 0
            
            if success:
                # Create base tags
                tags = job_data.get("document_tags") or []
                base_tags = tags.copy()
                if "github" not in base_tags:
                    base_tags.append("github")
                if "sdk" not in base_tags:
                    base_tags.append("sdk")
                
                # Process documentation files
                for doc in result.get("documents", []):
                    try:
                        # Skip if content is minimal
                        if not doc or len(doc.get("content", "")) < 50:
                            failures += 1
                            continue
                        
                        # Process content
                        title = doc.get("title", "Untitled Document")
                        content = doc.get("content", "")
                        
                        # Add metadata
                        metadata = doc.get("metadata", {})
                        metadata.update({
                            "scheduled_job_id": job_id,
                            "scheduled_job_name": job_data.get("name")
                        })
                        
                        # Create document
                        doc_obj = self.documents_service.create_document(
                            title=title,
                            content=content,
                            document_type=DocumentType.API_REFERENCE,
                            metadata=metadata,
                            tags=base_tags,
                            source_url=doc.get("file_url"),
                            enable_vector_embedding=True,
                            link_to_knowledge_graph=True
                        )
                        document_ids.append(doc_obj["id"])
                        doc_successes += 1
                    except Exception as e:
                        logger.error(f"Error creating document from GitHub file: {e}")
                        failures += 1
                
                # Process code examples
                for example in result.get("code_examples", []):
                    try:
                        if not example or len(example.get("content", "")) < 50:
                            continue
                        
                        # Add language tag
                        code_tags = base_tags.copy()
                        language = example.get("language", "")
                        if language:
                            code_tags.append(language)
                        code_tags.append("code-example")
                        
                        # Format content with markdown code block
                        title = example.get("title", "Untitled Example")
                        language_md = example.get("language", "")
                        description = example.get("description", "")
                        
                        # Create formatted content
                        content = f"# {title}\n\n"
                        if description:
                            content += f"{description}\n\n"
                        content += f"```{language_md}\n{example.get('content', '')}\n```"
                        
                        # Add metadata
                        metadata = {
                            "scheduled_job_id": job_id,
                            "scheduled_job_name": job_data.get("name"),
                            "repo_url": repo_url,
                            "file_path": example.get("file_path"),
                            "language": example.get("language"),
                            "type": "code_example"
                        }
                        
                        # Create document
                        doc_obj = self.documents_service.create_document(
                            title=title,
                            content=content,
                            document_type=DocumentType.CODE_EXAMPLE,
                            metadata=metadata,
                            tags=code_tags,
                            source_url=f"{repo_url.replace('.git', '')}/blob/main/{example.get('file_path', '')}",
                            enable_vector_embedding=True,
                            link_to_knowledge_graph=True
                        )
                        document_ids.append(doc_obj["id"])
                        code_successes += 1
                    except Exception as e:
                        logger.error(f"Error creating document from code example: {e}")
                        failures += 1
            
            # Update job status
            status = "success" if success and files_processed > 0 else "failed"
            
            details = {
                "repo_url": repo_url,
                "files_found": files_found,
                "files_processed": files_processed,
                "documentation_documents": doc_successes,
                "code_example_documents": code_successes,
                "failures": failures,
                "documents_created": doc_successes + code_successes,
                "document_ids": document_ids[:100]  # Limit for storage
            }
            
            self.update_job_status(job_id, status, details)
            logger.info(f"Completed scheduled GitHub scrape job {job_id}: {status} ({doc_successes + code_successes} documents created)")
            
        except Exception as e:
            logger.error(f"Error in scheduled GitHub scrape job {job_id}: {e}", exc_info=True)
            self.update_job_status(job_id, "error", {"error": str(e)})

    async def _run_verification_job(self, job_data: Dict[str, Any]):
        """Run a document verification job"""
        job_id = job_data.get("job_id")
        document_tags = job_data.get("document_tags", [])
        document_type = job_data.get("document_type")
        
        logger.info(f"Running scheduled verification job {job_id} for tags {document_tags}")
        
        try:
            # Get documents matching criteria
            query = {}
            if document_tags:
                query["tags"] = {"$in": document_tags}
            if document_type:
                query["document_type"] = document_type
                
            # Get documents (limited by max_documents)
            max_documents = job_data.get("max_documents", 1000)
            documents = self.documents_service.search_documents(
                query=query,
                limit=max_documents
            )
            
            # Track verification results
            verified_count = 0
            failed_count = 0
            repaired_count = 0
            verified_urls = []
            failed_urls = []
            repair_attempts = []
            
            # Process each document with source_url
            for doc in documents:
                source_url = doc.get("source_url")
                if not source_url:
                    continue
                    
                try:
                    # Verify document by re-scraping the source URL
                    verification_result = await self.scraper_service.scrape_url(source_url)
                    
                    if verification_result.get("success", False):
                        # Get new content
                        new_content = verification_result.get("content", "")
                        
                        # Check if content is substantial
                        if len(new_content) < 100:
                            failed_urls.append(source_url)
                            failed_count += 1
                            continue
                            
                        # Compare with existing content (simple length check for now)
                        existing_content = self.documents_service.get_document_content(doc["id"])
                        
                        # If content differs significantly, update the document
                        if len(new_content) > len(existing_content) * 1.2 or len(new_content) < len(existing_content) * 0.8:
                            # Content has changed significantly - update document
                            update_result = self.documents_service.update_document(
                                doc_id=doc["id"],
                                content=new_content,
                                commit_message=f"Verification update from scheduled job {job_id}"
                            )
                            
                            repair_attempts.append({
                                "document_id": doc["id"],
                                "source_url": source_url,
                                "status": "updated"
                            })
                            repaired_count += 1
                        else:
                            # Content is stable
                            verified_urls.append(source_url)
                            verified_count += 1
                            
                    else:
                        # Verification failed
                        failed_urls.append(source_url)
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error verifying document {doc['id']} with URL {source_url}: {e}")
                    failed_urls.append(source_url)
                    failed_count += 1
            
            # Update job status
            total = verified_count + failed_count
            success_rate = 0
            if total > 0:
                success_rate = (verified_count / total) * 100
                
            status = "success" if success_rate >= 90 else "partial" if success_rate >= 70 else "failed"
            
            details = {
                "document_tags": document_tags,
                "document_type": document_type,
                "documents_checked": total,
                "verified_count": verified_count,
                "failed_count": failed_count,
                "repaired_count": repaired_count,
                "success_rate": success_rate,
                "verified_urls": verified_urls[:50],  # Limit for storage
                "failed_urls": failed_urls[:50],  # Limit for storage
                "repair_attempts": repair_attempts[:50]  # Limit for storage
            }
            
            self.update_job_status(job_id, status, details)
            
            # If failures are found, consider a follow-up job
            if failed_urls:
                # Create a follow-up attempt for failed URLs
                await self._process_verification_document_failures(job_id, job_data, failed_urls)
                
            logger.info(f"Completed scheduled verification job {job_id}: {status} ({success_rate:.2f}% success rate)")
            
        except Exception as e:
            logger.error(f"Error in scheduled verification job {job_id}: {e}", exc_info=True)
            self.update_job_status(job_id, "error", {"error": str(e)})

    async def _process_verification_failures(self, parent_job_id: str, job_data: Dict[str, Any], failed_urls: List[str]):
        """Process URLs that failed verification"""
        if not failed_urls:
            return
            
        logger.info(f"Processing {len(failed_urls)} failed URLs from job {parent_job_id}")
        
        # Create a recovery job to try again with just the failed URLs
        recovery_job_id = f"{parent_job_id}_recovery_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Process in batches to avoid overloading
        batch_size = 10
        for i in range(0, len(failed_urls), batch_size):
            batch = failed_urls[i:i+batch_size]
            
            # Add delay between batches
            if i > 0:
                await asyncio.sleep(5)
                
            try:
                # Simply try to scrape and process these URLs again
                results = await self.scraper_service.scrape_urls(batch)
                
                # Process results
                recovered = 0
                for result in results:
                    if result.get("success", False):
                        try:
                            # Skip if content is minimal
                            content = result.get("content", "")
                            if len(content) < 100:
                                continue
                                
                            # Determine document type based on site_type
                            document_type = DocumentType.WEBPAGE
                            if job_data.get("site_type") == "legislation":
                                document_type = DocumentType.LEGAL
                            elif job_data.get("site_type") == "hmrc":
                                document_type = DocumentType.ACCOUNTANCY
                            
                            # Add metadata
                            metadata = result.get("metadata", {})
                            metadata.update({
                                "scheduled_job_id": parent_job_id,
                                "recovery_job_id": recovery_job_id,
                                "crawl_timestamp": result.get("scraped_at")
                            })
                            
                            # Apply tags
                            tags = job_data.get("document_tags") or []
                            tags.append("recovery")
                            
                            # Create document
                            self.documents_service.create_document(
                                title=result["title"],
                                content=content,
                                document_type=document_type,
                                metadata=metadata,
                                tags=tags,
                                source_url=result["url"],
                                enable_vector_embedding=True,
                                link_to_knowledge_graph=True
                            )
                            recovered += 1
                        except Exception as e:
                            logger.error(f"Error creating recovery document for URL {result.get('url')}: {e}")
                
                logger.info(f"Recovery batch {i//batch_size + 1}: Processed {len(batch)} URLs, recovered {recovered}")
                
            except Exception as e:
                logger.error(f"Error processing recovery batch: {e}")
                
    async def _process_verification_document_failures(self, parent_job_id: str, job_data: Dict[str, Any], failed_urls: List[str]):
        """Process documents with URLs that failed verification"""
        if not failed_urls:
            return
            
        logger.info(f"Processing {len(failed_urls)} failed document URLs from verification job {parent_job_id}")
        
        # Create a recovery job ID
        recovery_job_id = f"{parent_job_id}_recovery_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Process in batches to avoid overloading
        batch_size = 10
        for i in range(0, len(failed_urls), batch_size):
            batch = failed_urls[i:i+batch_size]
            
            # Add delay between batches
            if i > 0:
                await asyncio.sleep(5)
                
            try:
                # Try to scrape these URLs again
                results = await self.scraper_service.scrape_urls(batch)
                
                # Process results
                recovered = 0
                for result in results:
                    if result.get("success", False):
                        try:
                            # Skip if content is minimal
                            content = result.get("content", "")
                            if len(content) < 100:
                                continue
                                
                            # Find existing document with this source_url
                            query = {"source_url": result["url"]}
                            docs = self.documents_service.search_documents(query=query, limit=1)
                            
                            if docs:
                                # Update existing document
                                self.documents_service.update_document(
                                    doc_id=docs[0]["id"],
                                    content=content,
                                    commit_message=f"Verification recovery from job {recovery_job_id}"
                                )
                            else:
                                # Create new document
                                document_type = DocumentType.WEBPAGE
                                tags = job_data.get("document_tags") or []
                                tags.append("recovery")
                                
                                # Add metadata
                                metadata = result.get("metadata", {})
                                metadata.update({
                                    "verification_job_id": parent_job_id,
                                    "recovery_job_id": recovery_job_id
                                })
                                
                                self.documents_service.create_document(
                                    title=result["title"],
                                    content=content,
                                    document_type=document_type,
                                    metadata=metadata,
                                    tags=tags,
                                    source_url=result["url"],
                                    enable_vector_embedding=True,
                                    link_to_knowledge_graph=True
                                )
                                
                            recovered += 1
                        except Exception as e:
                            logger.error(f"Error processing recovery document for URL {result.get('url')}: {e}")
                
                logger.info(f"Document recovery batch {i//batch_size + 1}: Processed {len(batch)} URLs, recovered {recovered}")
                
            except Exception as e:
                logger.error(f"Error processing document recovery batch: {e}")