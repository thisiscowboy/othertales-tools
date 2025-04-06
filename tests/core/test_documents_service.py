import pytest
import os
import shutil
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.documents_service import DocumentsService
from app.models.documents import DocumentType

@pytest.fixture
def temp_dir():
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def documents_service(temp_dir):
    # Mock dependencies
    with patch('app.core.documents_service.get_config') as mock_config, \
         patch('app.core.documents_service.GitService') as mock_git_service:
        
        # Set up GitService mock
        git_service_instance = MagicMock()
        mock_git_service.return_value = git_service_instance
        
        # Configure service
        service = DocumentsService(base_path=temp_dir)
        
        # Ensure get_status always returns clean repo status
        git_service_instance.get_status.return_value = {
            "branch": "main",
            "clean": True,
            "untracked": [],
            "modified": [],
            "staged": []
        }
        
        yield service

class TestDocumentsService:
    def test_create_document(self, documents_service, temp_dir):
        # Test creating a document
        doc = documents_service.create_document(
            title="Test Document",
            content="This is test content.",
            document_type=DocumentType.GENERIC,
            metadata={"source": "test"},
            tags=["test", "document"]
        )
        
        # Verify document was created
        assert doc["title"] == "Test Document"
        assert doc["document_type"] == DocumentType.GENERIC.value
        assert "source" in doc["metadata"]
        assert "test" in doc["tags"]
        
        # Check that document ID was generated
        assert doc["id"] is not None
        
        # Verify file exists
        doc_path = Path(temp_dir) / DocumentType.GENERIC.value / f"{doc['id']}.md"
        assert doc_path.exists()
        
        # Verify content
        content = doc_path.read_text()
        assert "Test Document" in content
        assert "This is test content." in content
        
    def test_get_document(self, documents_service):
        # Create a document
        doc = documents_service.create_document(
            title="Get Test",
            content="Content for retrieval test",
            document_type=DocumentType.GENERIC
        )
        
        # Get the document
        retrieved = documents_service.get_document(doc["id"])
        
        # Verify retrieval
        assert retrieved["id"] == doc["id"]
        assert retrieved["title"] == "Get Test"
        assert "Content for retrieval" in retrieved["content_preview"]
        
    def test_update_document(self, documents_service):
        # Create a document
        doc = documents_service.create_document(
            title="Original Title",
            content="Original content",
            document_type=DocumentType.GENERIC,
            tags=["original"]
        )
        
        # Update the document
        updated = documents_service.update_document(
            doc_id=doc["id"],
            title="Updated Title",
            content="Updated content",
            tags=["updated", "document"]
        )
        
        # Verify update
        assert updated["title"] == "Updated Title"
        assert "updated" in updated["tags"]
        
        # Check the document content
        content = documents_service.get_document_content(doc["id"])
        assert content["content"] == "Updated content"
        
    def test_delete_document(self, documents_service):
        # Create a document
        doc = documents_service.create_document(
            title="To Delete",
            content="This document will be deleted",
            document_type=DocumentType.GENERIC
        )
        
        # Delete the document
        result = documents_service.delete_document(doc["id"])
        assert result is True
        
        # Verify document is gone
        assert documents_service.get_document(doc["id"]) is None
        
        # Verify file is removed
        doc_path = Path(documents_service.base_path) / DocumentType.GENERIC.value / f"{doc['id']}.md"
        assert not doc_path.exists()
        
    def test_search_documents(self, documents_service):
        # Create test documents
        docs = []
        # Document 1
        docs.append(documents_service.create_document(
            title="Search Test One",
            content="This document has specific content to find.",
            document_type=DocumentType.GENERIC,
            tags=["search", "test"]
        ))
        
        # Document 2
        docs.append(documents_service.create_document(
            title="Search Test Two",
            content="Another document with different content.",
            document_type=DocumentType.GENERIC,
            tags=["search", "different"]
        ))
        
        # Document 3 (with different type)
        docs.append(documents_service.create_document(
            title="Different Type",
            content="This has a different document type.",
            document_type=DocumentType.WEBPAGE,
            tags=["webpage"]
        ))
        
        # Search by content
        results = documents_service.search_documents("specific content")
        assert len(results) == 1
        assert results[0]["id"] == docs[0]["id"]
        
        # Search by tag
        results = documents_service.search_documents("", tags=["search"])
        assert len(results) == 2
        
        # Search by document type
        results = documents_service.search_documents("", doc_type=DocumentType.WEBPAGE.value)
        assert len(results) == 1
        assert results[0]["id"] == docs[2]["id"]
        
    def test_get_document_versions(self, documents_service):
        # Create a document
        doc = documents_service.create_document(
            title="Version Test",
            content="Initial version",
            document_type=DocumentType.GENERIC
        )
        
        # Mock git log response
        documents_service.git_service.get_log.return_value = {
            "commits": [
                {
                    "hash": "abc123",
                    "message": "Updated document",
                    "author": "Test User",
                    "date": "2023-01-02 10:00:00"
                },
                {
                    "hash": "def456",
                    "message": "Created document",
                    "author": "Test User",
                    "date": "2023-01-01 10:00:00"
                }
            ]
        }
        
        # Get document versions
        versions = documents_service.get_document_versions(doc["id"])
        
        # Verify versions
        assert len(versions) == 2
        assert versions[0]["hash"] == "abc123"
        assert versions[1]["hash"] == "def456"
        assert versions[1]["message"] == "Created document"
        
    @patch('app.core.documents_service.markdown')
    @patch('weasyprint.HTML')
    def test_convert_document_format(self, mock_weasyprint, mock_markdown, documents_service):
        # Create a document
        doc = documents_service.create_document(
            title="Convert Format Test",
            content="# Heading\nContent to convert",
            document_type=DocumentType.GENERIC
        )
        
        # Mock markdown conversion
        mock_markdown.markdown.return_value = "<h1>Heading</h1><p>Content to convert</p>"
        
        # Mock PDF generation
        mock_pdf = MagicMock()
        mock_pdf.write_pdf.return_value = b"PDF content"
        mock_weasyprint.return_value = mock_pdf
        
        # Convert document to PDF
        result = documents_service.convert_document_format(doc["id"], "pdf")
        
        # Verify conversion
        assert mock_markdown.markdown.called
        assert mock_weasyprint.called
        assert result == b"PDF content"
