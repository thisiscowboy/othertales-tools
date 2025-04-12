import os
import pathlib
import logging
import hashlib
from typing import Optional, List, Dict, Any
import time
import threading
import glob
import shutil
import fnmatch
from app.utils.config import get_config

# Third-party imports in try/except for graceful handling
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    boto3 = None
    HAS_BOTO3 = False

logger = logging.getLogger(__name__)

class FilesystemService:
    def __init__(self):
        config = get_config()
        self.allowed_directories = [str(pathlib.Path(os.path.expanduser(d)).resolve())
                                   for d in config.allowed_directories]
        self.s3_client = None
        self.s3_resource = None
        
        if config.s3_access_key and config.s3_secret_key:
            try:
                if not HAS_BOTO3 or boto3 is None:
                    logger.warning("boto3 is not installed. S3 functionality will be disabled.")
                else:
                    self.s3_client = boto3.client(
                        's3',
                        aws_access_key_id=config.s3_access_key,
                        aws_secret_access_key=config.s3_secret_key,
                        region_name=config.s3_region
                    )
                    self.s3_resource = boto3.resource(
                        's3',
                        aws_access_key_id=config.s3_access_key,
                        aws_secret_access_key=config.s3_secret_key,
                        region_name=config.s3_region
                    )
                    logger.info("S3 client initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize S3 client: %s", e)
                
        self.cache_enabled = hasattr(config, 'file_cache_enabled') and config.file_cache_enabled
        self.cache_dir = pathlib.Path("./cache")
        self.cache_max_age = 3600
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("File caching enabled")

    def normalize_path(self, requested_path: str) -> pathlib.Path:
        if not requested_path:
            raise ValueError("Empty path not allowed")
            
        # Resolve the path to handle symlinks
        requested = pathlib.Path(os.path.expanduser(requested_path)).resolve()
        
        # Check if the resolved path is within the allowed directories
        for allowed in self.allowed_directories:
            allowed_path = pathlib.Path(allowed).resolve()
            if str(requested).startswith(str(allowed_path)):
                return requested
                
        raise ValueError(f"Access denied: {requested} is outside allowed directories.")

    def _cache_key(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        key_parts = [storage, path]
        if bucket:
            key_parts.append(bucket)
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def read_file(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        if storage == "local":
            file_path = self.normalize_path(path)
            return file_path.read_text(encoding="utf-8")
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            response = self.s3_client.get_object(Bucket=bucket, Key=path)
            return response['Body'].read().decode('utf-8')
        else:
            raise ValueError(f"Unsupported storage type: {storage}")

    def read_file_cached(self, path: str, max_age: Optional[int] = None,
                        storage: str = "local", bucket: Optional[str] = None) -> str:
        if not self.cache_enabled:
            return self.read_file(path, storage, bucket)
            
        max_age = max_age or self.cache_max_age
        cache_key = self._cache_key(path, storage, bucket)
        
        with self.cache_lock:
            cache_entry = self.cache.get(cache_key)
            if cache_entry:
                content, timestamp = cache_entry
                if time.time() - timestamp <= max_age:
                    return content
                    
            if self.cache_dir.exists():
                cache_file = self.cache_dir / cache_key
                if cache_file.exists():
                    file_age = time.time() - cache_file.stat().st_mtime
                    if file_age <= max_age:
                        content = cache_file.read_text(encoding="utf-8")
                        self.cache[cache_key] = (content, time.time())
                        return content
        
        # Cache miss or stale cache
        content = self.read_file(path, storage, bucket)
        
        # Update cache
        with self.cache_lock:
            self.cache[cache_key] = (content, time.time())
            if self.cache_dir.exists():
                cache_file = self.cache_dir / cache_key
                cache_file.write_text(content, encoding="utf-8")
                
        return content

    def write_file(self, path: str, content: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        if storage == "local":
            file_path = self.normalize_path(path)
            # Ensure the directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully written to {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            self.s3_client.put_object(
                Bucket=bucket,
                Key=path,
                Body=content.encode("utf-8"),
                ContentType="text/plain"
            )
            return f"Successfully written to s3://{bucket}/{path}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")

    def create_directory(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        if storage == "local":
            dir_path = self.normalize_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory: {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            # S3 doesn't have directories, but we can create an empty marker
            self.s3_client.put_object(
                Bucket=bucket,
                Key=f"{path.rstrip('/')}/.keep",
                Body=b"",
                ContentType="application/x-empty"
            )
            return f"Successfully created directory: s3://{bucket}/{path}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")

    def delete_file(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        if storage == "local":
            file_path = self.normalize_path(path)
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                    return f"Successfully deleted file: {path}"
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    return f"Successfully deleted directory: {path}"
            else:
                return f"File or directory does not exist: {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            self.s3_client.delete_object(Bucket=bucket, Key=path)
            return f"Successfully deleted s3://{bucket}/{path}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")

    def search_files(self, directory: str, pattern: str, storage: str = "local", bucket: Optional[str] = None) -> List[Dict[str, Any]]:
        if storage == "local":
            dir_path = self.normalize_path(directory)
            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Directory does not exist: {directory}")
            
            matches = []
            for root, _, _ in os.walk(dir_path):
                for file in glob.glob(os.path.join(root, pattern)):
                    file_path = pathlib.Path(file)
                    stat = file_path.stat()
                    matches.append({
                        "path": file,
                        "name": file_path.name,
                        "size": stat.st_size,
                        "modified": int(stat.st_mtime)
                    })
            return matches
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
                
            matches = []
            prefix = directory.rstrip("/") + "/"
            for obj in self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix).get('Contents', []):
                key = obj['Key']
                if fnmatch.fnmatch(key, pattern):
                    matches.append({
                        "path": key,
                        "name": pathlib.Path(key).name,
                        "size": obj['Size'],
                        "modified": int(obj['LastModified'].timestamp())
                    })
            return matches
        else:
            raise ValueError(f"Unsupported storage type: {storage}")

    def list_directory(self, directory: str, storage: str = "local", bucket: Optional[str] = None,
                      recursive: bool = False) -> Dict[str, Any]:
        if storage == "local":
            dir_path = self.normalize_path(directory)
            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"Directory does not exist: {directory}")
                
            items = []
            if recursive:
                for root, dirs, files in os.walk(dir_path):
                    for d in dirs:
                        d_path = pathlib.Path(os.path.join(root, d))
                        rel_path = str(d_path.relative_to(dir_path))
                        items.append({
                            "name": d,
                            "path": rel_path,
                            "type": "directory",
                            "size": None,
                            "last_modified": None
                        })
                    for f in files:
                        f_path = pathlib.Path(os.path.join(root, f))
                        rel_path = str(f_path.relative_to(dir_path))
                        stat = f_path.stat()
                        items.append({
                            "name": f,
                            "path": rel_path,
                            "type": "file",
                            "size": stat.st_size,
                            "last_modified": int(stat.st_mtime)
                        })
            else:
                for item in dir_path.iterdir():
                    rel_path = str(item.relative_to(dir_path))
                    if item.is_dir():
                        items.append({
                            "name": item.name,
                            "path": rel_path,
                            "type": "directory",
                            "size": None,
                            "last_modified": None
                        })
                    elif item.is_file():
                        stat = item.stat()
                        items.append({
                            "name": item.name,
                            "path": rel_path,
                            "type": "file",
                            "size": stat.st_size,
                            "last_modified": int(stat.st_mtime)
                        })
                        
            return {
                "path": str(directory),
                "items": items
            }
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
                
            # Normalize directory path for S3
            prefix = directory
            if not prefix.endswith('/') and prefix:
                prefix = prefix + '/'
                
            # Empty prefix means root directory
            if prefix == '/':
                prefix = ''
                
            items = []
            directories = set()
            
            if recursive:
                # For recursive, list all objects and extract directories
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key == prefix:
                            continue
                            
                        if key.endswith('/'):
                            directories.add(key)
                        else:
                            rel_path = key[len(prefix):]
                            items.append({
                                "name": os.path.basename(key),
                                "path": rel_path,
                                "type": "file",
                                "size": obj['Size'],
                                "last_modified": int(obj['LastModified'].timestamp())
                            })
                            
                        # Extract intermediate directories
                        parts = key[len(prefix):].split('/')
                        for i in range(len(parts) - 1):
                            d = prefix + '/'.join(parts[:i+1]) + '/'
                            directories.add(d)
                            
                # Add all directories
                for d in directories:
                    rel_path = d[len(prefix):]
                    items.append({
                        "name": os.path.basename(d.rstrip('/')),
                        "path": rel_path,
                        "type": "directory",
                        "size": None,
                        "last_modified": None
                    })
            else:
                # For non-recursive, use a delimiter to get directories
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
                    # Add subdirectories
                    for common_prefix in page.get('CommonPrefixes', []):
                        d = common_prefix['Prefix']
                        rel_path = d[len(prefix):]
                        name = rel_path.rstrip('/')
                        items.append({
                            "name": name,
                            "path": rel_path,
                            "type": "directory",
                            "size": None,
                            "last_modified": None
                        })
                        
                    # Add files
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key == prefix:
                            continue
                        rel_path = key[len(prefix):]
                        if not rel_path or '/' in rel_path:
                            continue
                        items.append({
                            "name": os.path.basename(key),
                            "path": rel_path,
                            "type": "file",
                            "size": obj['Size'],
                            "last_modified": int(obj['LastModified'].timestamp())
                        })
                        
            return {
                "path": directory,
                "items": sorted(items, key=lambda x: x["name"])
            }
        else:
            raise ValueError(f"Unsupported storage type: {storage}")

    def file_exists(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> bool:
        if storage == "local":
            try:
                file_path = self.normalize_path(path)
                return file_path.exists()
            except Exception:
                return False
        elif storage == "s3":
            if not bucket or not self.s3_client:
                return False
            try:
                self.s3_client.head_object(Bucket=bucket, Key=path)
                return True
            except Exception:
                return False
        else:
            return False

    def invalidate_cache(self, path: Optional[str] = None, storage: str = "local",
                         bucket: Optional[str] = None) -> None:
        if not self.cache_enabled:
            return
            
        with self.cache_lock:
            if path:
                cache_key = self._cache_key(path, storage, bucket)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                cache_file = self.cache_dir / cache_key
                if cache_file.exists():
                    cache_file.unlink()
            else:
                self.cache.clear()
                for cache_file in self.cache_dir.glob("*"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning("Failed to delete cache file %s: %s", cache_file, e)