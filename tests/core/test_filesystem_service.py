import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.core.filesystem_service import FilesystemService


@pytest.fixture
def test_temp_dir():
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def fs_service(test_temp_dir):
    # Configure service with test paths
    with patch("app.core.filesystem_service.get_config") as mock_config:
        mock_config.return_value.allowed_directories = [test_temp_dir]
        mock_config.return_value.file_cache_enabled = False
        mock_config.return_value.s3_access_key = None
        mock_config.return_value.s3_secret_key = None
        service = FilesystemService()
        yield service


class TestFilesystemService:
    def test_normalize_path(self, fs_service, test_temp_dir):
        # Test path normalization
        test_path = os.path.join(test_temp_dir, "test_file.txt")
        normalized = fs_service.normalize_path(test_path)
        assert str(normalized) == str(Path(test_path).resolve())

    def test_read_write_file(self, fs_service, test_temp_dir):
        # Test writing and reading a file
        test_path = os.path.join(test_temp_dir, "test_file.txt")
        content = "Test content"

        # Write file
        result = fs_service.write_file(test_path, content)
        assert "Successfully wrote to" in result

        # Read file
        read_content = fs_service.read_file(test_path)
        assert read_content == content

    def test_write_read_binary_file(self, fs_service, test_temp_dir):
        # Test writing and reading a binary file
        test_path = os.path.join(test_temp_dir, "test_binary.bin")
        content = b"\x00\x01\x02\x03"

        # Write binary file
        result = fs_service.write_file_binary(test_path, content)
        assert "Successfully wrote to" in result

        # Read binary file
        read_content = fs_service.read_file_binary(test_path)
        assert read_content == content

    def test_create_list_directory(self, fs_service, test_temp_dir):
        # Test creating and listing directories
        test_dir = os.path.join(test_temp_dir, "test_dir")

        # Create directory
        result = fs_service.create_directory(test_dir)
        assert "Successfully created directory" in result
        assert os.path.isdir(test_dir)

        # Create a file in the directory
        test_file = os.path.join(test_dir, "test_file.txt")
        fs_service.write_file(test_file, "Test content")

        # List directory
        listing = fs_service.list_directory(test_dir)
        assert listing["path"] == test_dir
        assert len(listing["items"]) == 1
        assert listing["items"][0]["name"] == "test_file.txt"
        assert listing["items"][0]["type"] == "file"

    def test_delete_file(self, fs_service, test_temp_dir):
        # Test deleting a file
        test_file = os.path.join(test_temp_dir, "file_to_delete.txt")
        fs_service.write_file(test_file, "Delete me")

        # Verify file exists
        assert os.path.exists(test_file)

        # Delete file
        result = fs_service.delete_file(test_file)
        assert "Successfully deleted" in result
        assert not os.path.exists(test_file)

    def test_search_files(self, fs_service, test_temp_dir):
        # Create test files
        os.makedirs(os.path.join(test_temp_dir, "dir1"))
        os.makedirs(os.path.join(test_temp_dir, "dir2"))
        fs_service.write_file(os.path.join(test_temp_dir, "dir1", "file1.txt"), "content")
        fs_service.write_file(os.path.join(test_temp_dir, "dir2", "file2.txt"), "content")
        fs_service.write_file(os.path.join(test_temp_dir, "dir1", "file.md"), "markdown")

        # Search for txt files
        results = fs_service.search_files(test_temp_dir, "*.txt")
        assert len(results) == 2
        assert any("file1.txt" in r for r in results)
        assert any("file2.txt" in r for r in results)

        # Search with specific pattern
        results = fs_service.search_files(os.path.join(test_temp_dir, "dir1"), "*.md")
        assert len(results) == 1
        assert "file.md" in results[0]

    @patch("boto3.client")
    @patch("boto3.resource")
    def test_s3_integration(self, mock_s3_resource, mock_s3_client, test_temp_dir):
        # Mock the S3 client and resource
        mock_client = MagicMock()
        mock_resource = MagicMock()
        mock_s3_client.return_value = mock_client
        mock_s3_resource.return_value = mock_resource

        # Mock response for list_objects_v2
        mock_client.list_objects_v2.return_value = {
            "CommonPrefixes": [{"Prefix": "folder/"}],
            "Contents": [
                {
                    "Key": "test.txt",
                    "Size": 100,
                    "LastModified": MagicMock(timestamp=lambda: 1234567890),
                }
            ],
        }

        # Create service with S3 config
        with patch("app.core.filesystem_service.get_config") as mock_config:
            mock_config.return_value.allowed_directories = [test_temp_dir]
            mock_config.return_value.file_cache_enabled = False
            mock_config.return_value.s3_access_key = "test_key"
            mock_config.return_value.s3_secret_key = "test_secret"
            mock_config.return_value.s3_region = "us-east-1"

            service = FilesystemService()

            # Test S3 directory listing
            service.list_directory("", storage="s3", bucket="test-bucket")
            assert mock_client.list_objects_v2.called

            # Test S3 file writing
            service.write_file("test.txt", "content", storage="s3", bucket="test-bucket")
            mock_client.put_object.assert_called_once()

    def test_file_exists(self, fs_service, test_temp_dir):
        # Test file existence check
        test_file = os.path.join(test_temp_dir, "existing_file.txt")
        test_dir = os.path.join(test_temp_dir, "existing_dir")

        # Create file and directory
        fs_service.write_file(test_file, "I exist")
        os.makedirs(test_dir)

        # Check existing file
        assert fs_service.file_exists(test_file)
        assert fs_service.file_exists(test_dir)

        # Check non-existing file
        assert not fs_service.file_exists(os.path.join(test_temp_dir, "nonexistent.txt"))
