import os
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import shutil
import re
# Set up logger
logger = logging.getLogger(__name__)
class GitService:
    """Service for Git operations with thread-safe repository access"""
    def __init__(self, default_author_name: str = "OtherTales", default_author_email: str = "system@othertales.com"):
        """Initialize the Git service"""
        self.repo_locks = {}
        self.default_author_name = default_author_name
        self.default_author_email = default_author_email
        # Try importing gitpython - optional dependency
        try:
            import git
            self.git = git
        except ImportError:
            logger.error("GitPython is not installed. Git functionality will be limited.")
            self.git = None
    def _get_repo_lock(self, repo_path: str):
        """Get or create a lock for a specific repository path"""
        if repo_path not in self.repo_locks:
            self.repo_locks[repo_path] = threading.Lock()
        return self.repo_locks[repo_path]
    def _get_repo(self, repo_path: str):
        """Get Git repository, creating it if it doesn't exist"""
        if not self.git:
            raise ValueError("GitPython is not installed")
        if not os.path.exists(repo_path):
            os.makedirs(repo_path, exist_ok=True)
        try:
            repo = self.git.Repo(repo_path)
            return repo
        except self.git.InvalidGitRepositoryError:
            # Initialize new repository
            repo = self.git.Repo.init(repo_path)
            # Configure basic settings
            with repo.config_writer() as config:
                config.set_value("user", "name", self.default_author_name)
                config.set_value("user", "email", self.default_author_email)
            # Create initial commit
            if not os.path.exists(os.path.join(repo_path, '.gitignore')):
                with open(os.path.join(repo_path, '.gitignore'), 'w') as f:
                    f.write("*.swp\n*.bak\n*.tmp\n*.orig\n*~\n")
                repo.git.add('.gitignore')
                repo.git.commit('-m', 'Initial commit: Add .gitignore')
            return repo
        except Exception as e:
            logger.error(f"Error initializing repository: {e}")
            raise ValueError(f"Failed to initialize repository: {e}")
    def get_status(self, repo_path: str) -> Dict[str, Any]:
        """Get the status of a Git repository"""
        repo = self._get_repo(repo_path)
        
        try:
            current_branch = repo.active_branch.name
        except TypeError:
            # Handle detached HEAD state
            current_branch = "DETACHED_HEAD"
            
        # Get staged files
        staged_files = []
        try:
            staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        except Exception as e:
            logger.warning(f"Error getting staged files: {e}")
            
        # Get modified but unstaged files
        unstaged_files = []
        try:
            unstaged_files = [item.a_path for item in repo.index.diff(None)]
        except Exception as e:
            logger.warning(f"Error getting unstaged files: {e}")
            
        # Get untracked files
        untracked_files = []
        try:
            untracked_files = repo.untracked_files
        except Exception as e:
            logger.warning(f"Error getting untracked files: {e}")
            
        return {
            "clean": not (staged_files or unstaged_files or untracked_files),
            "current_branch": current_branch,
            "staged_files": staged_files,
            "unstaged_files": unstaged_files,
            "untracked_files": untracked_files
        }
    def get_diff(self, repo_path: str, file_path: Optional[str] = None, target: Optional[str] = None) -> str:
        """Get diff of changes"""
        repo = self._get_repo(repo_path)
        try:
            if file_path and target:
                return repo.git.diff(target, file_path)
            elif file_path:
                return repo.git.diff('HEAD', file_path)
            elif target:
                return repo.git.diff(target)
            else:
                return repo.git.diff()
        except Exception as e:
            logger.error(f"Error getting diff: {e}")
            raise ValueError(f"Failed to get diff: {e}")
    def add_files(self, repo_path: str, files: List[str]) -> str:
        """Stage files for commit"""
        with self._get_repo_lock(repo_path):
            repo = self._get_repo(repo_path)
            try:
                repo.git.add(files)
                return "Files staged successfully"
            except Exception as e:
                logger.error(f"Error adding files: {e}")
                raise ValueError(f"Failed to add files: {e}")
    def commit_changes(self, repo_path: str, message: str,
                      author_name: Optional[str] = None,
                      author_email: Optional[str] = None) -> str:
        """Commit staged changes"""
        with self._get_repo_lock(repo_path):
            repo = self._get_repo(repo_path)
            author_name = author_name or self.default_author_name
            author_email = author_email or self.default_author_email
            
            try:
                # Set author for this commit
                with repo.config_writer() as config:
                    config.set_value("user", "name", author_name)
                    config.set_value("user", "email", author_email)
                    
                # Commit changes
                commit = repo.index.commit(message)
                return f"Committed changes with hash {commit.hexsha}"
            except Exception as e:
                logger.error(f"Error committing changes: {e}")
                raise ValueError(f"Failed to commit changes: {e}")
    def reset_changes(self, repo_path: str) -> str:
        """Reset staged changes"""
        with self._get_repo_lock(repo_path):
            repo = self._get_repo(repo_path)
            try:
                repo.git.reset()
                return "All staged changes reset"
            except Exception as e:
                logger.error(f"Error resetting changes: {e}")
                raise ValueError(f"Failed to reset changes: {e}")
    def get_log(self, repo_path: str, max_count: int = 10, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Get commit log"""
        repo = self._get_repo(repo_path)
        try:
            commits_data = []
            if file_path:
                # Get log for specific file
                commits = list(repo.iter_commits(paths=file_path, max_count=max_count))
            else:
                # Get log for entire repo
                commits = list(repo.iter_commits(max_count=max_count))
            
            for commit in commits:
                commits_data.append({
                    "hash": commit.hexsha,
                    "author": f"{commit.author.name} <{commit.author.email}>",
                    "date": commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "message": commit.message.strip()
                })
            
            return {"commits": commits_data}
        except Exception as e:
            logger.error(f"Error getting log: {e}")
            raise ValueError(f"Failed to get log: {e}")
    def create_branch(self, repo_path: str, branch_name: str, base_branch: Optional[str] = None) -> str:
        """Create a new branch"""
        with self._get_repo_lock(repo_path):
            repo = self._get_repo(repo_path)
            try:
                # Use the specified base branch or current branch
                if base_branch:
                    # Check if base branch exists
                    if base_branch not in repo.refs:
                        raise ValueError(f"Base branch '{base_branch}' does not exist")
                    base = repo.refs[base_branch]
                else:
                    base = repo.active_branch
                    
                # Create new branch
                repo.create_head(branch_name, base)
                return f"Created branch '{branch_name}'"
            except Exception as e:
                logger.error(f"Error creating branch: {e}")
                raise ValueError(f"Failed to create branch: {e}")
    def checkout_branch(self, repo_path: str, branch_name: str, create: bool = False) -> str:
        """Checkout a branch"""
        with self._get_repo_lock(repo_path):
            repo = self._get_repo(repo_path)
            try:
                # Check if branch exists
                branch_exists = branch_name in repo.refs
                
                # Create branch if it doesn't exist
                if create and not branch_exists:
                    repo.create_head(branch_name)
                elif not branch_exists:
                    raise ValueError(f"Branch '{branch_name}' does not exist")
                    
                # Checkout the branch
                repo.git.checkout(branch_name)
                return f"Switched to branch '{branch_name}'"
            except Exception as e:
                logger.error(f"Error checking out branch: {e}")
                raise ValueError(f"Failed to checkout branch: {e}")
    def clone_repo(self, repo_url: str, local_path: str, auth_token: Optional[str] = None) -> str:
        """Clone a Git repository"""
        if not self.git:
            raise ValueError("GitPython is not installed")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            # If auth token is provided, modify the URL
            if auth_token:
                if repo_url.startswith("https://"):
                    parsed_url = repo_url.replace("https://", f"https://x-access-token:{auth_token}@")
                    self.git.Repo.clone_from(parsed_url, local_path)
                else:
                    # For SSH or other protocols, use standard clone
                    self.git.Repo.clone_from(repo_url, local_path)
            else:
                self.git.Repo.clone_from(repo_url, local_path)
                
            return f"Successfully cloned repository to '{local_path}'"
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise ValueError(f"Failed to clone repository: {e}")
    def get_file_content_at_version(self, repo_path: str, file_path: str, version: str) -> str:
        """Get file content at a specific Git version"""
        repo = self._get_repo(repo_path)
        try:
            return repo.git.show(f"{version}:{file_path}")
        except Exception as e:
            logger.error(f"Error getting file content at version: {e}")
            raise ValueError(f"Failed to get file content at version {version}: {e}")
    def get_diff(self, repo_path: str, file_path: str, from_version: str, to_version: str = "HEAD") -> str:
        """Get the differences between two versions of a file"""
        repo = self._get_repo(repo_path)
        try:
            return repo.git.diff(from_version, to_version, "--", file_path)
        except Exception as e:
            logger.error(f"Error getting diff between versions: {e}")
            raise ValueError(f"Failed to get diff between versions: {e}")