"""
Production-grade summarization module for Weez MCP AI Agent.

This module provides intelligent document summarization capabilities for files
stored across cloud platforms like Google Drive, OneDrive, Slack, and local storage.
Supports both file-based and query-based summarization with configurable detail levels.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from .search import search_documents_by_similarity
from utils.openai_client import summarize_chunks
from utils.openai_client import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Custom exception for summarization-related errors."""
    pass


def validate_intent(intent: Dict[str, Any]) -> None:
    """
    Validate the intent dictionary for summarization requirements.
    
    Args:
        intent: Dictionary containing user intent with required fields
        
    Raises:
        SummarizationError: If validation fails
    """
    if not isinstance(intent, dict):
        raise SummarizationError("Intent must be a dictionary")
    
    # Check if action is summarize
    action = intent.get("action")
    if action != "summarize":
        raise SummarizationError(f"Invalid action '{action}'. Expected 'summarize'")
    
    # Check if either file_id or query_text is present
    file_id = intent.get("file_id")
    query_text = intent.get("query_text")
    
    if not file_id and not query_text:
        raise SummarizationError("Either 'file_id' or 'query_text' must be provided")
    
    # Validate user_id is present
    user_id = intent.get("user_id")
    if not user_id:
        raise SummarizationError("'user_id' is required for summarization")
    
    logger.info(f"Intent validation passed for user {user_id}")


def determine_summarization_type(intent: Dict[str, Any]) -> str:
    """
    Determine the type of summarization based on the intent.
    
    Args:
        intent: Dictionary containing user intent
        
    Returns:
        String indicating summarization type: "file-based" or "query-based"
    """
    file_id = intent.get("file_id")
    query_text = intent.get("query_text")
    
    if file_id and not query_text:
        return "file-based"
    elif query_text and not file_id:
        return "query-based"
    elif file_id and query_text:
        # If both are present, prioritize query-based summarization
        logger.info("Both file_id and query_text present, using query-based summarization")
        return "query-based"
    else:
        # This should not happen due to validation, but handle it gracefully
        raise SummarizationError("Unable to determine summarization type")


def build_search_filters(intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build filters dictionary for search_documents_by_similarity based on intent.
    
    Args:
        intent: Dictionary containing user intent
        
    Returns:
        Dictionary containing filters or None if no filters needed
    """
    filters = {}
    
    # Add file_id filter if present
    if intent.get("file_id"):
        filters["file_id"] = intent["file_id"]
    
    # Add platform filter if present
    if intent.get("platform"):
        filters["platform"] = intent["platform"]
    
    # Add time_range filter if present
    if intent.get("time_range"):
        filters["time_range"] = intent["time_range"]
    
    # Return None if no filters, otherwise return the filters dict
    return filters if filters else None


def search_chunks_by_similarity(intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search for document chunks using similarity-based search.
    
    Args:
        intent: Dictionary containing user intent
        
    Returns:
        List of chunk dictionaries from search results
        
    Raises:
        SummarizationError: If search fails
    """
    user_id = intent["user_id"]
    summarization_type = determine_summarization_type(intent)
    
    # Build filters from intent
    filters = build_search_filters(intent)
    
    # For query-based summarization, use query_text to generate embedding
    if summarization_type == "query-based":
        query_text = intent["query_text"]
        logger.info(f"Generating embedding for query: {query_text}")
        
        try:
            embedding = get_embedding(query_text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise SummarizationError(f"Failed to generate embedding for query: {str(e)}")
    
    # For file-based summarization, use a generic query or file_id as query
    else:
        # Use a generic query for file-based summarization to retrieve all content
        generic_query = f"content from file {intent['file_id']}"
        logger.info(f"Generating embedding for file-based query: {generic_query}")
        
        try:
            embedding = get_embedding(generic_query)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise SummarizationError(f"Failed to generate embedding for file search: {str(e)}")
    
    # Set top_k based on summarization needs (more chunks for detailed summaries)
    summary_type = intent.get("summary_type", "short")
    top_k = 20 if summary_type == "detailed" else 10
    
    logger.info(f"Searching for documents with top_k={top_k}, filters={filters}")
    
    try:
        chunks = search_documents_by_similarity(
            embedding=embedding,
            user_id=user_id,
            filters=filters,
            top_k=top_k
        )
        
        logger.info(f"Found {len(chunks)} chunks from similarity search")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        raise SummarizationError(f"Failed to search documents: {str(e)}")


def extract_file_name(chunks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract file name from chunks metadata for file-based summarization.
    
    Args:
        chunks: List of chunk dictionaries containing metadata
        
    Returns:
        File name if found, None otherwise
    """
    if not chunks:
        return None
    
    # Try to extract file name from the first chunk's metadata
    first_chunk = chunks[0]
    
    # Check various possible metadata fields for file name
    metadata_fields = ["fileName", "file_name", "filename", "name", "title"]
    
    for field in metadata_fields:
        if field in first_chunk and first_chunk[field]:
            return first_chunk[field]
    
    # If no file name found in metadata, try to extract from file_id or other fields
    if "file_id" in first_chunk:
        return f"Document_{first_chunk['file_id']}"
    
    return None


def prepare_chunks_for_summary(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Prepare chunks for summarization by extracting text content.
    
    Args:
        chunks: List of chunk dictionaries from search results
        
    Returns:
        List of text strings ready for summarization
    """
    text_chunks = []
    
    for chunk in chunks:
        # Extract text content from various possible fields
        text_content = None
        
        # Try different field names for chunk content
        content_fields = ["content", "text", "chunk_text", "body", "data"]
        
        for field in content_fields:
            if field in chunk and chunk[field]:
                text_content = chunk[field]
                break
        
        if text_content:
            text_chunks.append(str(text_content))
        else:
            logger.warning(f"No text content found in chunk: {chunk}")
    
    return text_chunks


def summarize_document(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to perform document summarization based on user intent.
    
    Args:
        intent: Dictionary containing structured user intent with fields:
            - action: Must be "summarize"
            - file_id: Optional file identifier for file-based summarization
            - query_text: Optional query text for query-based summarization
            - summary_type: Optional summary detail level ("short" or "detailed")
            - user_id: Required user identifier
            - platform: Optional platform identifier
            - time_range: Optional time range filter
            
    Returns:
        Dictionary containing:
            - summary: Generated summary text
            - summary_type: Type of summary generated
            - chunks_used: Number of chunks used for summarization
            - fileName: File name (only for file-based summarization)
            
    Raises:
        SummarizationError: If summarization fails
    """
    try:
        # Validate input intent
        validate_intent(intent)
        
        # Set default summary type if not specified
        summary_type = intent.get("summary_type", "short")
        if summary_type not in ["short", "detailed"]:
            logger.warning(f"Invalid summary_type '{summary_type}', defaulting to 'short'")
            summary_type = "short"
        
        # Determine summarization type
        summarization_type = determine_summarization_type(intent)
        
        logger.info(f"Starting {summarization_type} summarization for user {intent['user_id']}")
        
        # Search for relevant chunks using similarity search
        try:
            chunks = search_chunks_by_similarity(intent)
        except Exception as e:
            logger.error(f"Error searching for chunks: {str(e)}")
            raise SummarizationError(f"Failed to retrieve document chunks: {str(e)}")
        
        # Check if any chunks were found
        if not chunks:
            if summarization_type == "file-based":
                error_msg = f"No content found for file_id: {intent.get('file_id')}"
            else:
                error_msg = f"No relevant content found for query: {intent.get('query_text')}"
            
            logger.warning(error_msg)
            raise SummarizationError(error_msg)
        
        logger.info(f"Found {len(chunks)} chunks for summarization")
        
        # Prepare chunks for summarization
        text_chunks = prepare_chunks_for_summary(chunks)
        
        if not text_chunks:
            raise SummarizationError("No valid text content found in retrieved chunks")
        
        # Generate summary using OpenAI
        try:
            query_for_summary = intent.get("query_text") if summarization_type == "query-based" else None
            summary = summarize_chunks(text_chunks, summary_type, query_for_summary)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")
        
        # Prepare response
        response = {
            "summary": summary,
            "summary_type": summary_type,
            "chunks_used": len(text_chunks)
        }
        
        # Add file name for file-based summarization
        if summarization_type == "file-based":
            file_name = extract_file_name(chunks)
            if file_name:
                response["fileName"] = file_name
        
        logger.info(f"Successfully generated {summary_type} summary using {len(text_chunks)} chunks")
        
        return response
        
    except SummarizationError:
        # Re-raise custom errors
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in summarization: {str(e)}")
        raise SummarizationError(f"Unexpected error during summarization: {str(e)}")


def summarize_file(file_id: str, user_id: str, summary_type: str = "short", 
                  platform: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for file-based summarization.
    
    Args:
        file_id: Identifier of the file to summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        
    Returns:
        Dictionary containing summarization results
    """
    intent = {
        "action": "summarize",
        "file_id": file_id,
        "user_id": user_id,
        "summary_type": summary_type
    }
    
    if platform:
        intent["platform"] = platform
    
    return summarize_document(intent)


def summarize_query(query_text: str, user_id: str, summary_type: str = "short",
                   platform: Optional[str] = None, time_range: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for query-based summarization.
    
    Args:
        query_text: Query text to search and summarize
        user_id: User identifier
        summary_type: Type of summary ("short" or "detailed")
        platform: Optional platform identifier
        time_range: Optional time range filter
        
    Returns:
        Dictionary containing summarization results
    """
    intent = {
        "action": "summarize",
        "query_text": query_text,
        "user_id": user_id,
        "summary_type": summary_type
    }
    
    if platform:
        intent["platform"] = platform
    
    if time_range:
        intent["time_range"] = time_range
    
    return summarize_document(intent)


if __name__ == "__main__":
    # Sample usage and testing
    
    # Example 1: File-based summarization
    print("=== File-based Summarization Test ===")
    try:
        file_intent = {
            "action": "summarize",
            "file_id": "doc_12345",
            "user_id": "user_67890",
            "summary_type": "short",
            "platform": "google_drive"
        }
        
        result = summarize_document(file_intent)
        print(f"File Summary Result: {result}")
        
    except SummarizationError as e:
        print(f"File summarization error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example 2: Query-based summarization
    print("\n=== Query-based Summarization Test ===")
    try:
        query_intent = {
            "action": "summarize",
            "query_text": "What are the main findings about AI in healthcare?",
            "user_id": "user_67890",
            "summary_type": "detailed",
            "platform": "all"
        }
        
        result = summarize_document(query_intent)
        print(f"Query Summary Result: {result}")
        
    except SummarizationError as e:
        print(f"Query summarization error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example 3: Using convenience functions
    print("\n=== Convenience Functions Test ===")
    try:
        # Test file summarization convenience function
        file_result = summarize_file("doc_98765", "user_12345", "short", "onedrive")
        print(f"Convenience file summary: {file_result}")
        
        # Test query summarization convenience function
        query_result = summarize_query("machine learning trends", "user_12345", "detailed")
        print(f"Convenience query summary: {query_result}")
        
    except Exception as e:
        print(f"Convenience function error: {e}")
    
    # Example 4: Error handling test
    print("\n=== Error Handling Test ===")
    try:
        invalid_intent = {
            "action": "invalid_action",
            "user_id": "user_12345"
        }
        
        result = summarize_document(invalid_intent)
        
    except SummarizationError as e:
        print(f"Expected error caught: {e}")
    
    print("\nAll tests completed!")
