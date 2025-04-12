import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Body, HTTPException, Path, Query
from app.models.documents import (
    DocumentType,
    CreateDocumentRequest,
    UpdateDocumentRequest,
    DocumentResponse,
    DocumentVersionResponse,
    DocumentContentResponse,
)
from app.core.documents_service import DocumentsService

# Set up logger
logger = logging.getLogger(__name__)
# Create router
router = APIRouter()
# Initialize documents service
documents_service = DocumentsService()


@router.post(
    "/create",
    response_model=DocumentResponse,
    summary="Create a new document",
    description="Create a new document with the specified content and metadata",
)
async def create_document(request: CreateDocumentRequest = Body(...)):
    """Create a new document with the specified content and metadata."""
    try:
        # Get global config settings
        from app.utils.config import get_config
        config = get_config()
        
        # Override global settings if explicitly specified in request
        embedding_enabled = request.embedding_enabled
        if not config.vector_embedding_enabled:
            embedding_enabled = False
            
        knowledge_graph_linking = request.knowledge_graph_linking
        if not config.knowledge_graph_auto_link:
            knowledge_graph_linking = False
        
        result = documents_service.create_document(
            title=request.title,
            content=request.content,
            document_type=request.document_type,
            metadata=request.metadata,
            tags=request.tags,
            source_url=request.source_url,
            storage_type=request.storage_type,
            # Pass the knowledge graph and embedding flags
            enable_vector_embedding=embedding_enabled,
            link_to_knowledge_graph=knowledge_graph_linking
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating document: {str(e)}")


@router.post(
    "/create/manuscript",
    response_model=DocumentResponse,
    summary="Create a new manuscript",
    description="Create a new manuscript (novel, book, story) with chapter structure",
)
async def create_manuscript(
    title: str = Body(..., description="Title of the manuscript"),
    summary: str = Body(..., description="Summary or synopsis of the manuscript"),
    structure: List[Dict[str, str]] = Body(..., description="Chapter structure (titles)"),
    tags: Optional[List[str]] = Body(None, description="Tags for the manuscript"),
):
    """Create a new manuscript document with chapter structure."""
    try:
        # Create content with chapter structure
        content = f"# {title}\n\n## Summary\n\n{summary}\n\n## Chapters\n\n"
        
        # Add chapter structure
        for i, chapter in enumerate(structure, 1):
            chapter_title = chapter.get("title", f"Chapter {i}")
            chapter_summary = chapter.get("summary", "")
            content += f"### Chapter {i}: {chapter_title}\n\n"
            if chapter_summary:
                content += f"{chapter_summary}\n\n"
        
        # Add tags for manuscript if not provided
        manuscript_tags = tags or []
        if "manuscript" not in manuscript_tags:
            manuscript_tags.append("manuscript")
        
        # Create the document
        result = documents_service.create_document(
            title=title,
            content=content,
            document_type=DocumentType.MANUSCRIPT,
            metadata={"type": "novel", "chapters": len(structure)},
            tags=manuscript_tags,
            enable_vector_embedding=True,
            link_to_knowledge_graph=True
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating manuscript: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating manuscript: {str(e)}")


@router.get(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
    description="Get document metadata by ID",
)
async def get_document(doc_id: str = Path(..., description="Document ID")):
    """Get document metadata by ID."""
    try:
        result = documents_service.get_document(doc_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")


@router.get(
    "/{doc_id}/content",
    response_model=DocumentContentResponse,
    summary="Get document content",
    description="Get full document content by ID",
)
async def get_document_content(
    doc_id: str = Path(..., description="Document ID"),
    version: Optional[str] = Query(None, description="Optional version to retrieve"),
):
    """Get document content by ID, optionally from a specific version."""
    try:
        result = documents_service.get_document_content(doc_id, version)
        if not result:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document content: {str(e)}")


@router.put(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Update document",
    description="Update an existing document",
)
async def update_document(
    doc_id: str = Path(..., description="Document ID"),
    request: UpdateDocumentRequest = Body(...),
):
    """Update an existing document."""
    try:
        result = documents_service.update_document(
            doc_id=doc_id,
            title=request.title,
            content=request.content,
            metadata=request.metadata,
            tags=request.tags,
            commit_message=request.commit_message,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")


@router.put(
    "/{doc_id}/chapter/{chapter_number}",
    response_model=DocumentResponse,
    summary="Update document chapter",
    description="Update a specific chapter in a manuscript document",
)
async def update_chapter(
    doc_id: str = Path(..., description="Document ID"),
    chapter_number: int = Path(..., ge=1, description="Chapter number"),
    title: Optional[str] = Body(None, description="New chapter title"),
    content: str = Body(..., description="New chapter content"),
    commit_message: str = Body("Updated chapter", description="Commit message"),
):
    """Update a specific chapter in a manuscript document."""
    try:
        # First get the document to ensure it exists and is a manuscript
        doc = documents_service.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
        if doc.get("document_type") != DocumentType.MANUSCRIPT:
            raise HTTPException(status_code=400, detail="Document is not a manuscript")
            
        # Get full document content
        doc_content = documents_service.get_document_content(doc_id)
        if not doc_content:
            raise HTTPException(status_code=404, detail=f"Document content not found")
            
        # Parse the document to find chapters
        lines = doc_content["content"].split("\n")
        chapters = []
        chapter_start_lines = []
        
        # Find chapter headings
        for i, line in enumerate(lines):
            if line.startswith("### Chapter "):
                chapters.append(line)
                chapter_start_lines.append(i)
        
        # Check if the requested chapter exists
        if chapter_number > len(chapters):
            raise HTTPException(status_code=404, detail=f"Chapter {chapter_number} not found")
            
        # Determine the start and end of the chapter
        chapter_start = chapter_start_lines[chapter_number - 1]
        chapter_end = chapter_start_lines[chapter_number] if chapter_number < len(chapters) else len(lines)
        
        # Update the chapter title if provided
        if title:
            chapter_title_parts = chapters[chapter_number - 1].split(":", 1)
            if len(chapter_title_parts) > 1:
                lines[chapter_start] = f"{chapter_title_parts[0]}: {title}"
            else:
                lines[chapter_start] = f"{chapters[chapter_number - 1]}: {title}"
                
        # Replace chapter content
        lines[chapter_start + 1:chapter_end] = ["", content, ""]
        
        # Rejoin the document
        updated_content = "\n".join(lines)
        
        # Update the document
        result = documents_service.update_document(
            doc_id=doc_id,
            content=updated_content,
            commit_message=f"{commit_message} - Chapter {chapter_number}",
        )
        
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating chapter: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating chapter: {str(e)}")


@router.delete(
    "/{doc_id}",
    summary="Delete document",
    description="Delete a document by ID",
)
async def delete_document(doc_id: str = Path(..., description="Document ID")):
    """Delete a document by ID."""
    try:
        success = documents_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return {"message": f"Document {doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.get(
    "/{doc_id}/versions",
    response_model=List[DocumentVersionResponse],
    summary="Get document versions",
    description="Get version history for a document",
)
async def get_document_versions(
    doc_id: str = Path(..., description="Document ID"),
    max_versions: int = Query(10, description="Maximum number of versions to return"),
):
    """Get version history for a document."""
    try:
        versions = documents_service.get_document_versions(doc_id, max_versions)
        return versions
    except Exception as e:
        logger.error(f"Error getting document versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document versions: {str(e)}")


@router.get(
    "/search",
    response_model=List[DocumentResponse],
    summary="Search documents",
    description="Search documents by query, type, and tags",
)
async def search_documents(
    query: Optional[str] = Query(None, description="Search query"),
    doc_type: Optional[str] = Query(None, description="Document type"),
    tags: Optional[List[str]] = Query(None, description="Tags to filter by"),
    limit: int = Query(10, description="Maximum number of results"),
):
    """Search documents by query, type, and tags."""
    try:
        results = documents_service.search_documents(query, doc_type, tags, limit)
        return results
    except Exception as e:
        logger.error(f"Error searching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@router.get(
    "/{doc_id}/diff",
    summary="Get document diff",
    description="Get differences between document versions",
)
async def get_document_diff(
    doc_id: str = Path(..., description="Document ID"),
    from_version: str = Query(..., description="Base version"),
    to_version: str = Query("HEAD", description="Target version"),
):
    """Get differences between document versions."""
    try:
        diff = documents_service.get_document_diff(doc_id, from_version, to_version)
        return diff
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting document diff: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document diff: {str(e)}")


@router.get(
    "/{doc_id}/export/{format}",
    summary="Export document",
    description="Export a document to a different format",
)
async def export_document(
    doc_id: str = Path(..., description="Document ID"),
    format: str = Path(..., description="Target format (pdf, docx)"),
):
    """Export a document to a different format."""
    try:
        if format.lower() not in ["pdf", "docx"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        document = documents_service.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        exported_bytes = documents_service.convert_document_format(doc_id, format.lower())
        
        from fastapi.responses import Response
        content_type = "application/pdf" if format.lower() == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        filename = f"{document['title'].replace(' ', '_')}.{format.lower()}"
        
        return Response(
            content=exported_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error exporting document: {str(e)}")


@router.post(
    "/{doc_id}/tag",
    response_model=DocumentResponse,
    summary="Add tags to document",
    description="Add tags to an existing document",
)
async def add_document_tags(
    doc_id: str = Path(..., description="Document ID"),
    tags: List[str] = Body(..., description="Tags to add"),
):
    """Add tags to an existing document."""
    try:
        document = documents_service.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        current_tags = document.get("tags", [])
        new_tags = list(set(current_tags + tags))
        
        result = documents_service.update_document(
            doc_id=doc_id,
            tags=new_tags,
            commit_message="Added tags",
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding tags: {str(e)}")


@router.post(
    "/restore-version/{doc_id}",
    response_model=DocumentResponse,
    summary="Restore document version",
    description="Restore a document to a previous version",
)
async def restore_document_version(
    doc_id: str = Path(..., description="Document ID"),
    version: str = Body(..., description="Version hash to restore"),
    commit_message: str = Body("Restored previous version", description="Commit message"),
):
    """Restore a document to a previous version."""
    try:
        # Get the content from the specified version
        version_content = documents_service.get_document_content(doc_id, version)
        if not version_content:
            raise HTTPException(status_code=404, detail=f"Version {version} of document {doc_id} not found")
        
        # Update the document with the content from the specified version
        result = documents_service.update_document(
            doc_id=doc_id,
            content=version_content["content"],
            commit_message=f"{commit_message} - Version {version[:7]}",
        )
        
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error restoring document version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error restoring document version: {str(e)}")


@router.post(
    "/semantic-search",
    response_model=List[DocumentResponse],
    summary="Semantic search",
    description="Search documents semantically using vector embeddings",
)
async def semantic_search(
    query: str = Body(..., embed=True, description="Search query"),
    limit: int = Query(5, description="Maximum number of results"),
):
    """Search documents semantically using vector embeddings."""
    try:
        results = documents_service.semantic_search(query, limit)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during semantic search: {str(e)}")