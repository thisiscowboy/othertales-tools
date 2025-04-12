from enum import Enum  # Added missing Enum import
from typing import List, Dict, Optional, Any  # Removed Union
from pydantic import BaseModel, Field  # Removed HttpUrl


class DocumentType(Enum):
    # Main document categories
    MANUSCRIPT = "manuscript"  # Novels, stories, books
    DOCUMENTATION = "documentation"  # SDK, API, technical docs
    ACCOUNTANCY = "accountancy"  # Tax, finance, accounting
    LEGAL = "legal"  # Law, regulation, contracts
    
    # Specific document types
    NOTES = "notes"
    OUTLINE = "outline"
    CHARACTER_SHEET = "character_sheet"
    WORLDBUILDING = "worldbuilding"
    RESEARCH = "research"
    GENERIC = "generic"
    WEBPAGE = "webpage"  # Web content
    
    # SDK/API specific types
    API_REFERENCE = "api_reference"
    SDK_GUIDE = "sdk_guide"
    CODE_EXAMPLE = "code_example"
    
    # Tax/Accounting specific types
    TAX_GUIDE = "tax_guide"
    ACCOUNTING_STANDARD = "accounting_standard"
    FINANCIAL_REPORT = "financial_report"
    
    # Legal specific types
    CONTRACT = "contract"
    REGULATION = "regulation"
    LEGAL_OPINION = "legal_opinion"
    CASE_LAW = "case_law"


class CreateDocumentRequest(BaseModel):
    """Request model for document creation"""

    title: str = Field(..., description="Document title", min_length=1)
    content: str = Field(..., description="Document content (text)")
    document_type: DocumentType = Field(DocumentType.GENERIC, description="Type of document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    storage_type: str = Field("local", description="Storage type (local or s3)")
    knowledge_graph_linking: bool = Field(True, description="Whether to automatically link to knowledge graph")
    embedding_enabled: bool = Field(True, description="Whether to generate vector embeddings for the document")


class UpdateDocumentRequest(BaseModel):
    """Request model for document updates"""

    title: Optional[str] = Field(None, description="Updated title")
    content: Optional[str] = Field(None, description="Updated content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    commit_message: str = Field("Updated document", description="Git commit message")


class DocumentResponse(BaseModel):
    """Response model for document operations"""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    document_type: DocumentType = Field(..., description="Type of document")
    created_at: int = Field(..., description="Creation timestamp")
    updated_at: int = Field(..., description="Last update timestamp")
    tags: List[str] = Field(..., description="Document tags")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    content_preview: str = Field(..., description="Preview of document content")
    size_bytes: int = Field(..., description="Document size in bytes")
    version_count: Optional[int] = Field(1, description="Number of versions")
    content_available: bool = Field(..., description="Whether full content is available")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    knowledge_graph_linked: bool = Field(False, description="Whether document is linked to knowledge graph")
    vector_embedding: bool = Field(False, description="Whether document has vector embeddings")
    related_entities: Optional[List[Dict[str, Any]]] = Field(None, description="Related entities from knowledge graph")


class DocumentVersionResponse(BaseModel):
    """Response model for document version history"""

    version_hash: str = Field(..., description="Git commit hash")
    commit_message: str = Field(..., description="Commit message")
    author: str = Field(..., description="Author name")
    timestamp: int = Field(..., description="Commit timestamp")
    changes: Optional[Dict[str, Any]] = Field(None, description="Changes in this version")


class DocumentContentResponse(BaseModel):
    """Response model for document content"""

    id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    version: Optional[str] = Field(None, description="Version hash if specific version requested")