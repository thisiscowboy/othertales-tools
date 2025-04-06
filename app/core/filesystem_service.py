import os
import pathlib
import difflib
import boto3
import logging
import hashlib
from typing import List, Optional, Dict, Union, Any, Generator, BinaryIO
from io import BytesIO
import time
import threading
from app.utils.config import get_config
# Set up logger
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
                logger.error(f"Failed to initialize S3 client: {e}")
        # Add caching with proper locking
        self.cache_enabled = hasattr(config, 'file_cache_enabled') and config.file_cache_enabled
        self.cache_dir = pathlib.Path("./cache")
        self.cache_max_age = 3600  # 1 hour
        self.cache = {}
        self.cache_lock = threading.Lock()
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("File caching enabled")
    def normalize_path(self, requested_path: str) -> pathlib.Path:
        """Normalize path and ensure it's within allowed directories"""
        # Handle special case of empty path
        if not requested_path:
            raise ValueError("Empty path not allowed")
        requested = pathlib.Path(os.path.expanduser(requested_path)).resolve()
        # Check if path is within any allowed directory
        for allowed in self.allowed_directories:
            if str(requested).startswith(allowed):
                return requested
        raise ValueError(f"Access denied: {requested} is outside allowed directories.")
    def _cache_key(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        """Generate a unique cache key for the given path"""
        key_parts = [storage, path]
        if bucket:
            key_parts.append(bucket)
        # Create a hash to ensure the key is safe for file system use
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    def read_file(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        """Read file content from local or S3 storage"""
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
    def read_file_cached(self, path: str, max_age: int = None,
                        storage: str = "local", bucket: Optional[str] = None) -> str:
        """Read file with caching"""
        if not self.cache_enabled:
            return self.read_file(path, storage, bucket)
        cache_key = self._cache_key(path, storage, bucket)
        max_age = max_age or self.cache_max_age
        # Check memory cache with proper locking
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if time.time() - entry["timestamp"] < max_age:
                    logger.debug(f"Cache hit for {path}")
                    return entry["content"]
        # Read actual file
        content = self.read_file(path, storage, bucket)
        # Update cache with proper locking
        with self.cache_lock:
            self.cache[cache_key] = {
                "content": content,
                "timestamp": time.time()
            }
            # Persist to disk cache as well
            cache_file = self.cache_dir / cache_key
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Failed to write to disk cache: {e}")
        return content
    def read_file_binary(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> bytes:
        """Read binary file content from local or S3 storage"""
        if storage == "local":
            file_path = self.normalize_path(path)
            if not file_path.exists():
                raise ValueError(f"File not found: {path}")
            if file_path.is_dir():
                raise ValueError(f"Path is a directory, not a file: {path}")
            return file_path.read_bytes()
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            response = self.s3_client.get_object(Bucket=bucket, Key=path)
            return response['Body'].read()
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def write_file(self, path: str, content: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        """Write content to a file in local or S3 storage"""
        if storage == "local":
            file_path = self.normalize_path(path)
            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            self.s3_client.put_object(
                Bucket=bucket,
                Key=path,
                Body=content.encode('utf-8'),
                ContentType='text/plain'
            )
            return f"Successfully wrote to s3://{bucket}/{path}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def write_file_binary(self, path: str, content: bytes,
                         storage: str = "local", bucket: Optional[str] = None) -> str:
        """Write binary content to a file in local or S3 storage"""
        if storage == "local":
            file_path = self.normalize_path(path)
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Write content to file
            with open(file_path, "wb") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            self.s3_client.put_object(Bucket=bucket, Key=path, Body=content)
            return f"Successfully wrote {len(content)} bytes to s3://{bucket}/{path}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def list_directory(self, path: str, storage: str = "local", bucket: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:
        """List contents of a directory in local or S3 storage"""
        if storage == "local":
            dir_path = self.normalize_path(path)
            if not dir_path.is_dir():
                raise ValueError("Provided path is not a directory")
            items = []
            if recursive:
                # Recursive listing
                for root, dirs, files in os.walk(dir_path):
                    root_path = pathlib.Path(root)
                    for name in dirs:
                        full_path = root_path / name
                        rel_path = full_path.relative_to(dir_path)
                        items.append({
                            "name": name,
                            "path": str(rel_path),
                            "type": "directory",
                            "size": None,
                            "last_modified": None
                        })
                    for name in files:
                        full_path = root_path / name
                        rel_path = full_path.relative_to(dir_path)
                        items.append({
                            "name": name,
                            "path": str(rel_path),
                            "type": "file",
                            "size": full_path.stat().st_size,
                            "last_modified": int(full_path.stat().st_mtime)
                        })
            else:
                # Non-recursive listing
                for entry in dir_path.iterdir():
                    is_dir = entry.is_dir()
                    items.append({
                        "name": entry.name,
                        "path": entry.name,
                        "type": "directory" if is_dir else "file",
                        "size": None if is_dir else entry.stat().st_size,
                        "last_modified": None if is_dir else int(entry.stat().st_mtime)
                    })
            return {
                "path": path,
                "items": items
            }
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            # Normalize S3 path - ensure it ends with / if it's not empty
            prefix = path
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            # List objects in the bucket with the given prefix
            if recursive:
                response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            else:
                response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            items = []
            # Process common prefixes (directories)
            for common_prefix in response.get('CommonPrefixes', []):
                prefix_path = common_prefix.get('Prefix', '')
                name = prefix_path.rstrip('/').split('/')[-1]
                items.append({
                    "name": name,
                    "path": prefix_path,
                    "type": "directory",
                    "size": None,
                    "last_modified": None
                })
            # Process objects (files)
            for obj in response.get('Contents', []):
                obj_key = obj.get('Key', '')
                # Skip the directory entry itself
                if obj_key == prefix:
                    continue
                name = obj_key.replace(prefix, '').split('/')[0]
                # Skip entries from subdirectories in non-recursive mode
                if not recursive and '/' in name:
                    continue
                items.append({
                    "name": name,
                    "path": obj_key,
                    "type": "file",
                    "size": obj.get('Size'),
                    "last_modified": int(obj.get('LastModified', 0).timestamp())
                })
            return {
                "path": path,
                "items": items
            }
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def create_directory(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        """Create a directory in local or S3 storage"""
        if storage == "local":
            dir_path = self.normalize_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            # S3 doesn't have directories, but we can create an empty object with a trailing slash
            key = path if path.endswith('/') else path + '/'
            self.s3_client.put_object(Bucket=bucket, Key=key, Body='')
            return f"Successfully created directory s3://{bucket}/{key}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def delete_file(self, path: str, storage: str = "local", bucket: Optional[str] = None) -> str:
        """Delete a file from local or S3 storage"""
        if storage == "local":
            file_path = self.normalize_path(path)
            if not file_path.exists():
                raise ValueError(f"File not found: {path}")
            if file_path.is_dir():
                raise ValueError(f"Path is a directory, not a file: {path}")
            file_path.unlink()
            return f"Successfully deleted {path}"
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            self.s3_client.delete_object(Bucket=bucket, Key=path)
            return f"Successfully deleted s3://{bucket}/{path}"
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def search_files(self, path: str, pattern: str, storage: str = "local",
                    bucket: Optional[str] = None, exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """Search for files matching a pattern"""
        exclude_patterns = exclude_patterns or []
        if storage == "local":
            base_path = self.normalize_path(path)
            results = []
            for root, dirs, files in os.walk(base_path):
                root_path = pathlib.Path(root)
                # Apply exclusion patterns to directories
                dirs[:] = [d for d in dirs if not any(pathlib.Path(root_path / d).match(exc) for exc in exclude_patterns)]
                for item in files:
                    item_path = root_path / item
                    # Skip excluded paths
                    if any(item_path.match(exc) for exc in exclude_patterns):
                        continue
                    # Check if pattern matches
                    if pattern.lower() in item.lower():
                        rel_path = item_path.relative_to(base_path)
                        results.append(str(rel_path))
            return results
        elif storage == "s3":
            if not bucket:
                raise ValueError("S3 bucket name is required for S3 storage")
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            # Normalize S3 path
            prefix = path
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            # List all objects with the prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            results = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj.get('Key')
                    filename = key.split('/')[-1]
                    # Skip excluded paths
                    if any(key.endswith(exc) for exc in exclude_patterns):
                        continue
                    # Check if pattern matches
                    if pattern.lower() in filename.lower():
                        # Return path relative to the search path
                        rel_path = key[len(prefix):] if key.startswith(prefix) else key
                        results.append(rel_path)
            return results
        else:
            raise ValueError(f"Unsupported storage type: {storage}")
    def read_large_file_chunked(self, path: str, chunk_size: int = 1024*1024) -> Generator[str, None, None]:
        """Read large file in chunks to avoid memory issues"""
        file_path = self.normalize_path(path)
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    def invalidate_cache(self, path: str = None,
                        storage: str = "local", bucket: Optional[str] = None):
        """Invalidate cache for a specific path or all paths"""
        if not self.cache_enabled:
            return
        with self.cache_lock:
            if path:
                # Invalidate specific path
                cache_key = self._cache_key(path, storage, bucket)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                # Remove from disk cache
                cache_file = self.cache_dir / cache_key
                if cache_file.exists():
                    cache_file.unlink()
            else:
                # Invalidate all cache
                self.cache.clear()
                # Clear disk cache
                for cache_file in self.cache_dir.glob("*"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")