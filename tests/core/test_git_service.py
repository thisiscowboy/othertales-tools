import pytest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.git_service import GitService

@pytest.fixture
def temp_dir():
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def git_service():
    service = GitService()
    yield service

@pytest.fixture
def git_repo(temp_dir, git_service):
    # Initialize a Git repository for testing
    repo_path = os.path.join(temp_dir, "test_repo")
    os.makedirs(repo_path)
    
    # Initialize git repo
    from git import Repo
    Repo.init(repo_path)
    
    # Create a test file and commit it
    test_file = os.path.join(repo_path, "test.txt")
    with open(test_file, "w") as f:
        f.write("Initial content")
    
    repo = Repo(repo_path)
    repo.git.add("test.txt")
    repo.git.config("user.email", "test@example.com")
    repo.git.config("user.name", "Test User")
    repo.git.commit("-m", "Initial commit")
    
    yield repo_path

class TestGitService:
    def test_get_status(self, git_service, git_repo):
        # Test getting repository status
        status = git_service.get_status(git_repo)
        
        assert "branch" in status
        assert status["clean"] is True
        assert len(status["untracked"]) == 0
        
    def test_add_and_commit(self, git_service, git_repo):
        # Create a new file
        new_file = os.path.join(git_repo, "new_file.txt")
        with open(new_file, "w") as f:
            f.write("New file content")
        
        # Add the file
        result = git_service.add_files(git_repo, ["new_file.txt"])
        assert "staged successfully" in result
        
        # Commit the file
        commit_result = git_service.commit_changes(
            git_repo, 
            "Test commit", 
            author_name="Test Author", 
            author_email="test@example.com"
        )
        
        assert "Committed changes with hash" in commit_result
        
        # Check status after commit
        status = git_service.get_status(git_repo)
        assert status["clean"] is True
        
    def test_get_log(self, git_service, git_repo):
        # Test getting commit log
        log = git_service.get_log(git_repo)
        
        assert "commits" in log
        assert len(log["commits"]) > 0
        assert "hash" in log["commits"][0]
        assert "message" in log["commits"][0]
        assert "Initial commit" in log["commits"][0]["message"]
        
    def test_create_checkout_branch(self, git_service, git_repo):
        # Create a new branch
        result = git_service.create_branch(git_repo, "test-branch")
        assert "Created branch" in result
        
        # Checkout the branch
        checkout_result = git_service.checkout_branch(git_repo, "test-branch")
        assert "Switched to branch" in checkout_result
        
        # Verify current branch
        status = git_service.get_status(git_repo)
        assert status["branch"] == "test-branch"
        
    def test_reset_changes(self, git_service, git_repo):
        # Create and stage a new file
        new_file = os.path.join(git_repo, "to_reset.txt")
        with open(new_file, "w") as f:
            f.write("Content to reset")
        
        git_service.add_files(git_repo, ["to_reset.txt"])
        
        # Reset changes
        result = git_service.reset_changes(git_repo)
        assert "reset" in result
        
        # Verify status
        status = git_service.get_status(git_repo)
        assert "to_reset.txt" in status["untracked"]
        
    def test_get_diff(self, git_service, git_repo):
        # Modify a file
        test_file = os.path.join(git_repo, "test.txt")
        with open(test_file, "w") as f:
            f.write("Initial content\nModified content")
        
        # Get diff
        diff = git_service.get_diff(git_repo)
        
        assert "diff" in diff
        assert "Modified content" in diff
        
    def test_remove_file(self, git_service, git_repo):
        # Test removing a file
        result = git_service.remove_file(git_repo, "test.txt")
        assert "removed" in result
        
        # Verify file is gone
        assert not os.path.exists(os.path.join(git_repo, "test.txt"))
        
    @patch('git.Repo.clone_from')
    def test_clone_repo(self, mock_clone, git_service, temp_dir):
        # Test cloning a repository
        clone_path = os.path.join(temp_dir, "cloned_repo")
        
        git_service.clone_repo("https://github.com/example/repo.git", clone_path)
        
        mock_clone.assert_called_once()
        assert "https://github.com/example/repo.git" in mock_clone.call_args[0]
        
    def test_batch_commit(self, git_service, git_repo):
        # Create multiple files
        files = []
        for i in range(3):
            file_path = os.path.join(git_repo, f"batch_file_{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Batch file {i} content")
            files.append(f"batch_file_{i}.txt")
        
        # Batch commit in groups
        file_groups = [files[0:2], files[2:]]
        result = git_service.batch_commit(git_repo, file_groups, "Batch commit")
        
        assert len(result) == 2  # Two commit hashes
        
        # Verify log
        log = git_service.get_log(git_repo)
        assert "Batch commit" in log["commits"][0]["message"]
