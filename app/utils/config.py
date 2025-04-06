import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import dotenv
# Load .env file if it exists
dotenv.load_dotenv()

LANGUAGE: python
SIZE: 2264 bytes
================================================================================
class Config(BaseSettings):
    # Server settings
    server_host: str = Field(os.getenv("SERVER_HOST", "0.0.0.0"), env="SERVER_HOST")
    server_port: int = Field(os.getenv("SERVER_PORT", 8000), env="SERVER_PORT")
    dev_mode: bool = Field(os.getenv("DEV_MODE", "False").lower() == "true", env="DEV_MODE")
    # Filesystem settings
    allowed_directories: List[str] = Field(default_factory=lambda: os.getenv("ALLOWED_DIRS", "./data").split(","))
    file_cache_enabled: bool = Field(os.getenv("FILE_CACHE_ENABLED", "False").lower() == "true", env="FILE_CACHE_ENABLED")
    # Memory settings
    memory_file_path: str = Field(os.getenv("MEMORY_FILE_PATH", "./data/memory.json"), env="MEMORY_FILE_PATH")
    use_graph_db: bool = Field(os.getenv("USE_GRAPH_DB", "False").lower() == "true", env="USE_GRAPH_DB")
    # Git settings
    default_git_username: str = Field(os.getenv("DEFAULT_COMMIT_USERNAME", "UnifiedTools"), env="DEFAULT_COMMIT_USERNAME")
    default_git_email: str = Field(os.getenv("DEFAULT_COMMIT_EMAIL", "tools@example.com"), env="DEFAULT_COMMIT_EMAIL")
    # S3 settings
    s3_access_key: Optional[str] = Field(os.getenv("S3_ACCESS_KEY"), env="S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = Field(os.getenv("S3_SECRET_KEY"), env="S3_SECRET_KEY")
    s3_region: Optional[str] = Field(os.getenv("S3_REGION"), env="S3_REGION")
    s3_bucket: Optional[str] = Field(os.getenv("S3_BUCKET"), env="S3_BUCKET")
    # Scraper settings
    scraper_min_delay: float = Field(float(os.getenv("SCRAPER_MIN_DELAY", "3")), env="SCRAPER_MIN_DELAY")
    scraper_max_delay: float = Field(float(os.getenv("SCRAPER_MAX_DELAY", "7")), env="SCRAPER_MAX_DELAY")
    user_agent: str = Field(os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; UnifiedToolsServer/1.0)"), env="USER_AGENT")
    # Set default data path
    scraper_data_path: str = Field(os.getenv("SCRAPER_DATA_PATH", "./data/scraped"), env="SCRAPER_DATA_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
_config = None
def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config