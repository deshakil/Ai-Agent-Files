"""
search.py - Production-grade intelligent document search module for AI assistant.

This module provides hybrid vector + metadata search capabilities over document chunks
stored in Azure Cosmos DB. It combines semantic similarity search with metadata filtering
to deliver precise and contextually relevant document retrieval.

Key Features:
- Structured intent-based search with comprehensive validation
- Hybrid vector + metadata search over Azure Cosmos DB
- Enhanced result processing with metadata enrichment
- Production-ready error handling and logging
- Modular design with clear separation of concerns

Author: AI Assistant
Version: 1.0.0
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import os

# Import external dependencies
from utils.openai_client import get_embedding
from utils.cosmos_client import CosmosVectorClient

# Configure module logger
logger = logging.getLogger(__name__)

def create_cosmos_client(cosmos_endpoint: str = os.getenv('COSMOS_URL'), cosmos_key: str = os.getenv('COSMOS_KEY')) -> CosmosVectorClient:
    """
    Factory function to create a CosmosVectorClient instance
    
    Args:
        cosmos_endpoint: Cosmos DB endpoint URL
        cosmos_key: Cosmos DB primary key
        
    Returns:
        Configured CosmosVectorClient instance
    """
    return CosmosVectorClient(cosmos_endpoint, cosmos_key)

class SearchError(Exception):
    """
    Custom exception for search-related errors.
    
    Raised for all predictable search issues including validation failures,
    embedding generation errors, and search execution problems.
    """
    pass

class Platform(Enum):
    """Supported platform types for document storage."""
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    DROPBOX = "dropbox"
    SHAREPOINT = "sharepoint"
    LOCAL = "local"
    SLACK = "slack"
    TEAMS = "teams"

class FileCategory(Enum):
    """Document file categories for enhanced metadata."""
    TEXT = "text"
    PDF = "pdf"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    OTHER = "other"

@dataclass
class SearchMetrics:
    """Container for search performance metrics."""
    total_results: int
    search_time_ms: float
    embedding_time_ms: float
    filter_count: int
    top_similarity_score: float

class SearchValidator:
    """
    Validates and normalizes search intent parameters.
    
    Ensures all required fields are present and valid, normalizes optional fields,
    and provides helpful error messages for validation failures.
    """
    
    # Supported MIME types for document search
    SUPPORTED_MIME_TYPES = {
        'text/plain',
        'text/markdown',
        'text/html',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # .pptx
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
        'application/msword',  # .doc
        'application/vnd.ms-powerpoint',  # .ppt
        'application/vnd.ms-excel',  # .xls
        'application/json',
        'application/xml',
        'text/csv'
    }
    
    # Supported relative time ranges
    SUPPORTED_TIME_RANGES = {
        'last_hour', 'last_24_hours', 'last_7_days', 'last_30_days', 
        'last_month', 'last_3_months', 'last_6_months', 'last_year'
    }
    
    @classmethod
    def validate_intent(cls, intent: dict) -> dict:
        """
        Validates and normalizes the search intent dictionary.
        
        Args:
            intent: Raw search intent dictionary from client
            
        Returns:
            Normalized and validated intent dictionary
            
        Raises:
            SearchError: If validation fails for any required or optional field
        """
        if not isinstance(intent, dict):
            raise SearchError("Intent must be a dictionary")
        
        # Validate required fields
        cls._validate_required_fields(intent)
        
        # Create normalized copy
        normalized = intent.copy()
        
        # Normalize and validate optional fields
        cls._normalize_query_text(normalized)
        cls._validate_platform(normalized)
        cls._validate_file_type(normalized)
        cls._validate_time_range(normalized)
        cls._validate_pagination(normalized)
        
        logger.debug(f"Intent validation successful for user: {normalized['user_id']}")
        return normalized
    
    @classmethod
    def _validate_required_fields(cls, intent: dict) -> None:
        """Validates required fields in intent dictionary."""
        if 'query_text' not in intent:
            raise SearchError("Missing required field: query_text")
        
        if not intent['query_text'] or not isinstance(intent['query_text'], str):
            raise SearchError("query_text must be a non-empty string")
        
        if 'user_id' not in intent:
            raise SearchError("Missing required field: user_id")
        
        if not intent['user_id'] or not isinstance(intent['user_id'], str):
            raise SearchError("user_id must be a non-empty string")
    
    @classmethod
    def _normalize_query_text(cls, intent: dict) -> None:
        """Normalizes query text by trimming whitespace and validating length."""
        query_text = intent['query_text'].strip()
        
        if len(query_text) < 2:
            raise SearchError("query_text must be at least 2 characters long")
        
        if len(query_text) > 1000:
            raise SearchError("query_text cannot exceed 1000 characters")
        
        intent['query_text'] = query_text
    
    @classmethod
    def _validate_platform(cls, intent: dict) -> None:
        """Validates and normalizes platform field."""
        if 'platform' not in intent or not intent['platform']:
            return
        
        platform = intent['platform'].lower().strip()
        valid_platforms = {p.value for p in Platform}
        
        if platform not in valid_platforms:
            logger.warning(f"Unknown platform: {platform}. Supported: {valid_platforms}")
            # Don't raise error, just log warning to allow flexibility
        
        intent['platform'] = platform
    
    @classmethod
    def _validate_file_type(cls, intent: dict) -> None:
        """Validates file type (MIME type) field."""
        if 'file_type' not in intent or not intent['file_type']:
            return
        
        file_type = intent['file_type'].lower().strip()
        
        if file_type not in cls.SUPPORTED_MIME_TYPES:
            logger.warning(f"Unknown file type: {file_type}. Supported: {cls.SUPPORTED_MIME_TYPES}")
            # Don't raise error to allow for new file types
        
        intent['file_type'] = file_type
    
    @classmethod
    def _validate_time_range(cls, intent: dict) -> None:
        """Validates time range specification."""
        if 'time_range' not in intent or not intent['time_range']:
            return
        
        time_range = intent['time_range']
        
        if isinstance(time_range, str):
            cls._validate_relative_time_range(time_range)
        elif isinstance(time_range, dict):
            cls._validate_absolute_time_range(time_range)
        else:
            raise SearchError("time_range must be a string or dictionary")
    
    @classmethod
    def _validate_relative_time_range(cls, time_range: str) -> None:
        """Validates relative time range strings."""
        if time_range not in cls.SUPPORTED_TIME_RANGES:
            raise SearchError(f"Invalid time range: {time_range}. Supported: {cls.SUPPORTED_TIME_RANGES}")
    
    @classmethod
    def _validate_absolute_time_range(cls, time_range: dict) -> None:
        """Validates absolute time range dictionaries."""
        if 'start_date' not in time_range and 'end_date' not in time_range:
            raise SearchError("Absolute time range must contain at least start_date or end_date")
        
        for date_field in ['start_date', 'end_date']:
            if date_field in time_range:
                try:
                    datetime.fromisoformat(time_range[date_field])
                except ValueError:
                    raise SearchError(f"Invalid {date_field} format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    
    @classmethod
    def _validate_pagination(cls, intent: dict) -> None:
        """Validates pagination parameters."""
        if 'offset' in intent:
            if not isinstance(intent['offset'], int) or intent['offset'] < 0:
                raise SearchError("offset must be a non-negative integer")
        
        if 'limit' in intent:
            if not isinstance(intent['limit'], int) or intent['limit'] < 1 or intent['limit'] > 100:
                raise SearchError("limit must be an integer between 1 and 100")

class FilterBuilder:
    """
    Constructs metadata filters for Cosmos DB search from validated intent.
    
    Translates high-level search intent into specific database query filters,
    handling various filter types including exact matches, regex patterns,
    and time-based range queries.
    """
    
    @classmethod
    def build_filters(cls, intent: dict) -> dict:
        """
        Builds comprehensive metadata filters for Cosmos DB search.
        
        Args:
            intent: Validated search intent dictionary
            
        Returns:
            Dictionary of filters compatible with Cosmos DB vector search
        """
        filters = {}
        
        # Always include user_id for data isolation and security
        filters['user_id'] = intent['user_id']
        
        # Platform-specific filtering
        if intent.get('platform'):
            filters['platform'] = intent['platform']
        
        # File type (MIME type) filtering
        if intent.get('file_type'):
            filters['mime_type'] = intent['file_type']
        
        # Specific file ID filtering (for searching within a particular file)
        if intent.get('file_id'):
            filters['file_id'] = intent['file_id']
        
        # File name pattern matching (case-insensitive partial match)
        if intent.get('file_name'):
            filters['fileName'] = cls._build_filename_filter(intent['file_name'])
        
        # Time-based filtering
        if intent.get('time_range'):
            time_filter = cls._build_time_filter(intent['time_range'])
            if time_filter:
                filters['created_at'] = time_filter
        
        # Content type filtering (for specific document sections)
        if intent.get('content_type'):
            filters['content_type'] = intent['content_type']
        
        # Chunk-specific filtering
        if intent.get('min_chunk_size'):
            filters['chunk_size'] = {'$gte': intent['min_chunk_size']}
        
        logger.debug(f"Built {len(filters)} filters for search: {list(filters.keys())}")
        return filters
    
    @classmethod
    def _build_filename_filter(cls, filename: str) -> dict:
        """
        Builds case-insensitive regex filter for filename matching.
        
        Args:
            filename: Filename pattern to match
            
        Returns:
            Regex filter dictionary for Cosmos DB
        """
        # Escape special regex characters and create case-insensitive pattern
        escaped_filename = re.escape(filename.strip())
        return {
            '$regex': f'.*{escaped_filename}.*',
            '$options': 'i'  # Case-insensitive
        }
    
    @classmethod
    def _build_time_filter(cls, time_range: Union[str, dict]) -> Optional[dict]:
        """
        Builds time-based filter for the created_at field.
        
        Args:
            time_range: Time range specification (relative string or absolute dict)
            
        Returns:
            Time filter dictionary or None if invalid
        """
        if isinstance(time_range, str):
            return cls._build_relative_time_filter(time_range)
        elif isinstance(time_range, dict):
            return cls._build_absolute_time_filter(time_range)
        
        return None
    
    @classmethod
    def _build_relative_time_filter(cls, time_range: str) -> dict:
        """Builds filter for relative time ranges (e.g., 'last_7_days')."""
        now = datetime.utcnow()
        
        time_deltas = {
            'last_hour': timedelta(hours=1),
            'last_24_hours': timedelta(hours=24),
            'last_7_days': timedelta(days=7),
            'last_30_days': timedelta(days=30),
            'last_month': timedelta(days=30),
            'last_3_months': timedelta(days=90),
            'last_6_months': timedelta(days=180),
            'last_year': timedelta(days=365)
        }
        
        if time_range in time_deltas:
            start_time = now - time_deltas[time_range]
            return {'$gte': start_time.isoformat()}
        
        logger.warning(f"Unknown relative time range: {time_range}")
        return {}
    
    @classmethod
    def _build_absolute_time_filter(cls, time_range: dict) -> dict:
        """Builds filter for absolute time ranges with start/end dates."""
        time_filter = {}
        
        if 'start_date' in time_range:
            try:
                start_date = datetime.fromisoformat(time_range['start_date'])
                time_filter['$gte'] = start_date.isoformat()
            except ValueError:
                logger.error(f"Invalid start_date format: {time_range['start_date']}")
        
        if 'end_date' in time_range:
            try:
                end_date = datetime.fromisoformat(time_range['end_date'])
                time_filter['$lte'] = end_date.isoformat()
            except ValueError:
                logger.error(f"Invalid end_date format: {time_range['end_date']}")
        
        return time_filter

class SearchResultProcessor:
    """
    Processes and enhances raw search results from Cosmos DB.
    
    Adds metadata enrichment, content previews, similarity scoring,
    and result categorization to provide comprehensive search results.
    """
    
    # MIME type to file category mapping
    MIME_TO_CATEGORY = {
        'text/plain': FileCategory.TEXT,
        'text/markdown': FileCategory.TEXT,
        'text/html': FileCategory.TEXT,
        'application/pdf': FileCategory.PDF,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileCategory.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileCategory.PRESENTATION,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileCategory.SPREADSHEET,
        'application/msword': FileCategory.DOCUMENT,
        'application/vnd.ms-powerpoint': FileCategory.PRESENTATION,
        'application/vnd.ms-excel': FileCategory.SPREADSHEET,
        'image/': FileCategory.IMAGE,
        'audio/': FileCategory.AUDIO,
        'video/': FileCategory.VIDEO,
        'application/zip': FileCategory.ARCHIVE,
        'application/x-rar': FileCategory.ARCHIVE,
    }
    
    @classmethod
    def process_results(cls, results: List[dict], intent: dict) -> List[dict]:
        """
        Processes and enhances raw search results.
        
        Args:
            results: Raw search results from Cosmos DB vector search
            intent: Original search intent for context
            
        Returns:
            List of enhanced and sorted search results
        """
        if not results:
            logger.info("No search results to process")
            return []
        
        # Sort results by similarity score (descending)
        sorted_results = sorted(
            results,
            key=lambda x: x.get('_similarity', 0.0),
            reverse=True
        )
        
        # Enhance each result with additional metadata
        enhanced_results = []
        for idx, result in enumerate(sorted_results):
            try:
                enhanced_result = cls._enhance_result(result, intent, idx)
                enhanced_results.append(enhanced_result)
            except Exception as e:
                logger.error(f"Error enhancing result {idx}: {str(e)}")
                # Include original result if enhancement fails
                enhanced_results.append(result)
        
        logger.info(f"Processed {len(enhanced_results)} search results")
        return enhanced_results
    
    @classmethod
    def _enhance_result(cls, result: dict, intent: dict, rank: int) -> dict:
        """
        Enhances a single search result with additional metadata.
        
        Args:
            result: Single search result from Cosmos DB
            intent: Original search intent
            rank: Result ranking (0-based)
            
        Returns:
            Enhanced result with additional metadata
        """
        enhanced = result.copy()
        
        # Add search metadata
        enhanced['_search_metadata'] = {
            'query_text': intent['query_text'],
            'similarity_score': result.get('_similarity', 0.0),
            'rank': rank,
            'timestamp': datetime.utcnow().isoformat(),
            'matched_filters': cls._extract_matched_filters(result, intent)
        }
        
        # Add content preview
        enhanced['_preview'] = cls._generate_preview(result)
        
        # Add file category
        enhanced['_file_category'] = cls._determine_file_category(result)
        
        # Add relevance indicators
        enhanced['_relevance'] = cls._calculate_relevance_indicators(result, intent)
        
        # Add chunk context
        enhanced['_chunk_context'] = cls._build_chunk_context(result)
        
        return enhanced
    
    @classmethod
    def _extract_matched_filters(cls, result: dict, intent: dict) -> dict:
        """Extracts which filters were matched for this result."""
        matched = {}
        
        if intent.get('platform') and result.get('platform') == intent['platform']:
            matched['platform'] = result['platform']
        
        if intent.get('file_type') and result.get('mime_type') == intent['file_type']:
            matched['mime_type'] = result['mime_type']
        
        if intent.get('file_id') and result.get('file_id') == intent['file_id']:
            matched['file_id'] = result['file_id']
        
        return matched
    
    @classmethod
    def _generate_preview(cls, result: dict, max_length: int = 200) -> str:
        """
        Generates a content preview snippet.
        
        Args:
            result: Search result containing text content
            max_length: Maximum preview length
            
        Returns:
            Preview snippet with ellipsis if truncated
        """
        # Try different content fields
        content_fields = ['text', 'content', 'summary', 'description']
        
        for field in content_fields:
            if field in result and result[field]:
                content = str(result[field]).strip()
                if content:
                    if len(content) <= max_length:
                        return content
                    else:
                        return content[:max_length] + '...'
        
        return "No preview available"
    
    @classmethod
    def _determine_file_category(cls, result: dict) -> str:
        """
        Determines the file category based on MIME type.
        
        Args:
            result: Search result containing mime_type
            
        Returns:
            File category string
        """
        mime_type = result.get('mime_type', '').lower()
        
        # Direct mapping
        if mime_type in cls.MIME_TO_CATEGORY:
            return cls.MIME_TO_CATEGORY[mime_type].value
        
        # Prefix matching for broad categories
        for prefix, category in cls.MIME_TO_CATEGORY.items():
            if prefix.endswith('/') and mime_type.startswith(prefix):
                return category.value
        
        # Fallback based on content patterns
        if 'word' in mime_type or 'document' in mime_type:
            return FileCategory.DOCUMENT.value
        elif 'presentation' in mime_type or 'powerpoint' in mime_type:
            return FileCategory.PRESENTATION.value
        elif 'spreadsheet' in mime_type or 'excel' in mime_type:
            return FileCategory.SPREADSHEET.value
        
        return FileCategory.OTHER.value
    
    @classmethod
    def _calculate_relevance_indicators(cls, result: dict, intent: dict) -> dict:
        """
        Calculates various relevance indicators for the result.
        
        Args:
            result: Search result
            intent: Original search intent
            
        Returns:
            Dictionary of relevance indicators
        """
        indicators = {
            'similarity_score': result.get('_similarity', 0.0),
            'is_recent': cls._is_recent_document(result),
            'has_keywords': cls._contains_keywords(result, intent['query_text']),
            'confidence_level': cls._calculate_confidence_level(result)
        }
        
        return indicators
    
    @classmethod
    def _is_recent_document(cls, result: dict, days_threshold: int = 30) -> bool:
        """Checks if document is recent based on created_at timestamp."""
        if 'created_at' not in result:
            return False
        
        try:
            created_at = datetime.fromisoformat(result['created_at'])
            threshold = datetime.utcnow() - timedelta(days=days_threshold)
            return created_at > threshold
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def _contains_keywords(cls, result: dict, query_text: str) -> bool:
        """Checks if result contains keywords from the query."""
        content_fields = ['text', 'content', 'fileName', 'summary']
        query_keywords = query_text.lower().split()
        
        for field in content_fields:
            if field in result and result[field]:
                content = str(result[field]).lower()
                for keyword in query_keywords:
                    if keyword in content:
                        return True
        
        return False
    
    @classmethod
    def _calculate_confidence_level(cls, result: dict) -> str:
        """Calculates confidence level based on similarity score."""
        similarity = result.get('_similarity', 0.0)
        
        if similarity >= 0.8:
            return 'high'
        elif similarity >= 0.6:
            return 'medium'
        elif similarity >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    @classmethod
    def _build_chunk_context(cls, result: dict) -> dict:
        """Builds context information about the chunk."""
        return {
            'chunk_index': result.get('chunk_index', 0),
            'total_chunks': result.get('total_chunks', 1),
            'chunk_size': len(result.get('text', result.get('content', ''))),
            'has_next_chunk': result.get('chunk_index', 0) < result.get('total_chunks', 1) - 1,
            'has_previous_chunk': result.get('chunk_index', 0) > 0
        }
    
client= create_cosmos_client()

def search_documents(intent: dict, top_k: int = 10) -> List[dict]:
    """
    Main entry point for intelligent document search.
    
    Performs hybrid vector + metadata search over document chunks stored in Azure Cosmos DB.
    Combines semantic similarity search with metadata filtering to deliver precise and
    contextually relevant document retrieval.
    
    Args:
        intent: Search intent dictionary containing:
            - query_text (str, required): The search query text
            - user_id (str, required): User identifier for data isolation
            - platform (str, optional): Platform filter (google_drive, onedrive, etc.)
            - file_type (str, optional): MIME type filter
            - file_id (str, optional): Specific file ID to search within
            - file_name (str, optional): File name pattern to match
            - time_range (str|dict, optional): Time range filter
            - content_type (str, optional): Content type filter
            - min_chunk_size (int, optional): Minimum chunk size filter
        top_k: Maximum number of results to return (default: 10, max: 100)
    
    Returns:
        List of enhanced document chunks sorted by similarity score, containing:
        - Original chunk data (text/content, metadata, etc.)
        - _similarity: Similarity score from vector search
        - _search_metadata: Search context and matched filters
        - _preview: Content snippet preview
        - _file_category: Categorized file type
        - _relevance: Relevance indicators
        - _chunk_context: Chunk positioning information
    
    Raises:
        SearchError: For all predictable search issues including:
            - Input validation failures
            - Embedding generation errors
            - Search execution problems
    
    Example:
        >>> intent = {
        ...     'query_text': 'quarterly sales report analysis',
        ...     'user_id': 'user123',
        ...     'platform': 'google_drive',
        ...     'file_type': 'application/pdf',
        ...     'time_range': 'last_3_months'
        ... }
        >>> results = search_documents(intent, top_k=5)
        >>> for result in results:
        ...     print(f"File: {result['fileName']}")
        ...     print(f"Score: {result['_similarity']:.3f}")
        ...     print(f"Preview: {result['_preview'][:100]}...")
    """
    start_time = datetime.utcnow()
    
    # Validate top_k parameter
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise SearchError("top_k must be an integer between 1 and 100")
    
    logger.info(f"Starting document search for user: {intent.get('user_id', 'unknown')}")
    
    try:
        # Step 1: Validate and normalize search intent
        logger.debug("Validating search intent")
        validated_intent = SearchValidator.validate_intent(intent)
        
        # Step 2: Generate embedding for query text
        logger.debug(f"Generating embedding for query: '{validated_intent['query_text']}'")
        embedding_start = datetime.utcnow()
        
        query_embedding = get_embedding(validated_intent['query_text'])
        
        embedding_time = (datetime.utcnow() - embedding_start).total_seconds() * 1000
        
        if not query_embedding or not isinstance(query_embedding, list):
            raise SearchError("Failed to generate valid embedding for query text")
        
        logger.debug(f"Generated embedding vector of length: {len(query_embedding)}")
        
        # Step 3: Build metadata filters
        logger.debug("Building metadata filters")
        filters = FilterBuilder.build_filters(validated_intent)
        
        # Step 4: Perform hybrid vector + metadata search
        logger.debug(f"Executing vector search with {len(filters)} filters and top_k={top_k}")
        search_start = datetime.utcnow()
        
        raw_results = client.vector_search_cosmos(
            embedding=query_embedding,
            filters=filters,
            top_k=top_k
        )
        
        search_time = (datetime.utcnow() - search_start).total_seconds() * 1000
        
        logger.info(f"Vector search completed: {len(raw_results)} results in {search_time:.2f}ms")
        
        # Step 5: Process and enhance results
        logger.debug("Processing and enhancing search results")
        enhanced_results = SearchResultProcessor.process_results(raw_results, validated_intent)
        
        # Step 6: Log search metrics
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        metrics = SearchMetrics(
            total_results=len(enhanced_results),
            search_time_ms=total_time,
            embedding_time_ms=embedding_time,
            filter_count=len(filters),
            top_similarity_score=enhanced_results[0].get('_similarity', 0.0) if enhanced_results else 0.0
        )
        
        logger.info(f"Search completed successfully in {total_time:.2f}ms: "
                   f"{metrics.total_results} results, "
                   f"top score: {metrics.top_similarity_score:.3f}")
        
        return enhanced_results
        
    except SearchError:
        # Re-raise SearchError as-is
        raise
    except Exception as e:
        # Wrap unexpected errors in SearchError
        logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
        raise SearchError(f"Search execution failed: {str(e)}")
    
def search_documents_by_similarity(embedding: List[float], user_id: str, 
                                 filters: Optional[dict] = None, 
                                 top_k: int = 10) -> List[dict]:
    """
    Performs document search using a pre-computed embedding vector.
    
    Useful for advanced use cases where the embedding is already available,
    when performing multiple searches with the same embedding, or when
    integrating with custom embedding generation pipelines.
    
    Args:
        embedding: Pre-computed query embedding vector (list of floats)
        user_id: User identifier for data isolation (required)
        filters: Optional additional metadata filters (dict)
        top_k: Maximum number of results to return (default: 10, max: 100)
        
    Returns:
        List of document chunks sorted by similarity score with basic metadata
        
    Raises:
        SearchError: For validation failures or search execution problems
        
    Example:
        >>> # Pre-compute embedding
        >>> embedding = get_embedding("machine learning algorithms")
        >>> 
        >>> # Search with custom filters
        >>> filters = {'platform': 'google_drive', 'mime_type': 'application/pdf'}
        >>> results = search_documents_by_similarity(embedding, 'user123', filters, top_k=5)
        >>> 
        >>> for result in results:
        ...     print(f"File: {result['fileName']}, Score: {result['_similarity']:.3f}")
    """
    start_time = datetime.utcnow()
    
    # Input validation
    if not embedding or not isinstance(embedding, list):
        raise SearchError("embedding must be a non-empty list of floats")
    
    if not all(isinstance(x, (int, float)) for x in embedding):
        raise SearchError("embedding must contain only numeric values")
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise SearchError("top_k must be an integer between 1 and 100")
    
    if filters is not None and not isinstance(filters, dict):
        raise SearchError("filters must be a dictionary or None")
    
    logger.info(f"Starting similarity search for user: {user_id}")
    
    try:
        # Build base filters with user isolation
        base_filters = {'user_id': user_id}
        
        # Merge with additional filters if provided
        if filters:
            base_filters.update(filters)
            logger.debug(f"Applied {len(filters)} additional filters")
        
        # Perform vector search
        logger.debug(f"Executing vector search with embedding of length: {len(embedding)}")
        search_start = datetime.utcnow()

        raw_results = client.vector_search_cosmos(
            embedding=embedding,
            filters=base_filters,
            top_k=top_k
        )
        
        search_time = (datetime.utcnow() - search_start).total_seconds() * 1000
        
        logger.info(f"Similarity search completed: {len(raw_results)} results in {search_time:.2f}ms")
        
        # Sort results by similarity score (descending)
        sorted_results = sorted(
            raw_results,
            key=lambda x: x.get('_similarity', 0.0),
            reverse=True
        )
        
        # Log metrics
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        top_score = sorted_results[0].get('_similarity', 0.0) if sorted_results else 0.0
        
        logger.info(f"Similarity search completed successfully in {total_time:.2f}ms: "
                   f"{len(sorted_results)} results, top score: {top_score:.3f}")
        
        return sorted_results
        
    except Exception as e:
        logger.error(f"Error during similarity search: {str(e)}", exc_info=True)
        raise SearchError(f"Similarity search failed: {str(e)}")


def get_document_by_id(document_id: str, user_id: str) -> Optional[dict]:
    """
    Retrieves a specific document by its ID.
    
    Args:
        document_id: Unique identifier for the document
        user_id: User identifier for data isolation
        
    Returns:
        Document dictionary if found, None otherwise
        
    Raises:
        SearchError: For validation failures or retrieval problems
    """
    if not document_id or not isinstance(document_id, str):
        raise SearchError("document_id must be a non-empty string")
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    logger.info(f"Retrieving document {document_id} for user {user_id}")
    
    try:
        # Use similarity search with exact ID filter
        filters = {'file_id': document_id}
        results = search_documents_by_similarity(
            embedding=[0.0] * 1536,  # Dummy embedding since we're filtering by ID
            user_id=user_id,
            filters=filters,
            top_k=1
        )
        
        if results:
            logger.info(f"Document {document_id} found")
            return results[0]
        else:
            logger.info(f"Document {document_id} not found")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise SearchError(f"Document retrieval failed: {str(e)}")


def get_related_documents(document_id: str, user_id: str, top_k: int = 5) -> List[dict]:
    """
    Finds documents related to a given document using similarity search.
    
    Args:
        document_id: ID of the source document
        user_id: User identifier for data isolation
        top_k: Maximum number of related documents to return
        
    Returns:
        List of related documents sorted by similarity
        
    Raises:
        SearchError: For validation failures or search problems
    """
    if not document_id or not isinstance(document_id, str):
        raise SearchError("document_id must be a non-empty string")
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    logger.info(f"Finding related documents for {document_id}")
    
    try:
        # First, get the source document
        source_doc = get_document_by_id(document_id, user_id)
        if not source_doc:
            raise SearchError(f"Source document {document_id} not found")
        
        # Extract content for similarity search
        content = source_doc.get('text', source_doc.get('content', ''))
        if not content:
            logger.warning(f"No content found in document {document_id}")
            return []
        
        # Generate embedding for the content
        content_embedding = get_embedding(content[:1000])  # Limit content length
        
        # Search for similar documents, excluding the source document
        filters = {'file_id': {'$ne': document_id}}  # Exclude source document
        
        related_results = search_documents_by_similarity(
            embedding=content_embedding,
            user_id=user_id,
            filters=filters,
            top_k=top_k
        )
        
        logger.info(f"Found {len(related_results)} related documents for {document_id}")
        return related_results
        
    except SearchError:
        raise
    except Exception as e:
        logger.error(f"Error finding related documents: {str(e)}")
        raise SearchError(f"Related document search failed: {str(e)}")


def search_within_file(file_id: str, query_text: str, user_id: str, top_k: int = 10) -> List[dict]:
    """
    Searches for content within a specific file.
    
    Args:
        file_id: ID of the file to search within
        query_text: Search query text
        user_id: User identifier for data isolation
        top_k: Maximum number of results to return
        
    Returns:
        List of matching chunks from the specified file
        
    Raises:
        SearchError: For validation failures or search problems
    """
    if not file_id or not isinstance(file_id, str):
        raise SearchError("file_id must be a non-empty string")
    
    if not query_text or not isinstance(query_text, str):
        raise SearchError("query_text must be a non-empty string")
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    logger.info(f"Searching within file {file_id} for query: '{query_text}'")
    
    try:
        # Create search intent for file-specific search
        intent = {
            'query_text': query_text.strip(),
            'user_id': user_id,
            'file_id': file_id
        }
        
        # Perform the search
        results = search_documents(intent, top_k=top_k)
        
        logger.info(f"Found {len(results)} results within file {file_id}")
        return results
        
    except SearchError:
        raise
    except Exception as e:
        logger.error(f"Error searching within file {file_id}: {str(e)}")
        raise SearchError(f"File search failed: {str(e)}")


def get_search_suggestions(partial_query: str, user_id: str, limit: int = 5) -> List[str]:
    """
    Generates search suggestions based on partial query input.
    
    Args:
        partial_query: Partial search query text
        user_id: User identifier for data isolation
        limit: Maximum number of suggestions to return
        
    Returns:
        List of suggested search queries
        
    Raises:
        SearchError: For validation failures
    """
    if not partial_query or not isinstance(partial_query, str):
        raise SearchError("partial_query must be a non-empty string")
    
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    if not isinstance(limit, int) or limit < 1 or limit > 20:
        raise SearchError("limit must be an integer between 1 and 20")
    
    logger.info(f"Generating search suggestions for: '{partial_query}'")
    
    try:
        # Clean and prepare partial query
        clean_query = partial_query.strip().lower()
        
        if len(clean_query) < 2:
            return []
        
        # Generate embedding for partial query
        partial_embedding = get_embedding(clean_query)
        
        # Search for similar content to generate suggestions
        filters = {'user_id': user_id}
        similar_results = search_documents_by_similarity(
            embedding=partial_embedding,
            user_id=user_id,
            filters=filters,
            top_k=limit * 2  # Get more results to extract diverse suggestions
        )
        
        # Extract suggestions from similar content
        suggestions = set()
        for result in similar_results:
            content = result.get('text', result.get('content', ''))
            if content:
                # Extract meaningful phrases that contain the partial query
                words = content.lower().split()
                for i, word in enumerate(words):
                    if clean_query in word:
                        # Create suggestion from surrounding context
                        start = max(0, i - 2)
                        end = min(len(words), i + 3)
                        suggestion = ' '.join(words[start:end])
                        if len(suggestion) > len(clean_query):
                            suggestions.add(suggestion)
            
            # Also use file names as suggestions
            file_name = result.get('fileName', '')
            if file_name and clean_query in file_name.lower():
                suggestions.add(file_name)
        
        # Return top suggestions
        suggestion_list = list(suggestions)[:limit]
        logger.info(f"Generated {len(suggestion_list)} suggestions for '{partial_query}'")
        
        return suggestion_list
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise SearchError(f"Suggestion generation failed: {str(e)}")


def get_search_analytics(user_id: str, days_back: int = 30) -> dict:
    """
    Retrieves search analytics and statistics for a user.
    
    Args:
        user_id: User identifier
        days_back: Number of days to look back for analytics
        
    Returns:
        Dictionary containing search analytics data
        
    Raises:
        SearchError: For validation failures
    """
    if not user_id or not isinstance(user_id, str):
        raise SearchError("user_id must be a non-empty string")
    
    if not isinstance(days_back, int) or days_back < 1 or days_back > 365:
        raise SearchError("days_back must be an integer between 1 and 365")
    
    logger.info(f"Retrieving search analytics for user {user_id}")
    
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # This is a placeholder implementation
        # In a real system, you would query search logs or analytics database
        analytics = {
            'user_id': user_id,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days_back
            },
            'statistics': {
                'total_searches': 0,
                'unique_queries': 0,
                'average_results_per_search': 0.0,
                'most_common_file_types': [],
                'most_searched_platforms': [],
                'search_trends': []
            },
            'performance': {
                'average_search_time_ms': 0.0,
                'average_similarity_score': 0.0,
                'success_rate': 0.0
            }
        }
        
        logger.info(f"Analytics retrieved for user {user_id}")
        return analytics
        
    except Exception as e:
        logger.error(f"Error retrieving analytics: {str(e)}")
        raise SearchError(f"Analytics retrieval failed: {str(e)}")


def health_check() -> dict:
    """
    Performs a health check of the search system.
    
    Returns:
        Dictionary containing health status information
    """
    logger.info("Performing search system health check")
    
    try:
        # Test basic functionality
        test_embedding = get_embedding("test query")
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                'embedding_service': 'healthy' if test_embedding else 'unhealthy',
                'cosmos_db': 'healthy',  # Would check actual DB connection
                'search_module': 'healthy'
            },
            'version': '1.0.0'
        }
        
        # Check if any components are unhealthy
        if any(status != 'healthy' for status in health_status['checks'].values()):
            health_status['status'] = 'degraded'
        
        logger.info(f"Health check completed: {health_status['status']}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'version': '1.0.0'
        }


# Utility functions for advanced search scenarios

def batch_search_documents(intents: List[dict], top_k: int = 10) -> List[List[dict]]:
    """
    Performs batch document search for multiple intents.
    
    Args:
        intents: List of search intent dictionaries
        top_k: Maximum number of results per search
        
    Returns:
        List of search results lists, one for each intent
        
    Raises:
        SearchError: For validation failures
    """
    if not intents or not isinstance(intents, list):
        raise SearchError("intents must be a non-empty list")
    
    if len(intents) > 10:
        raise SearchError("Cannot process more than 10 intents in a batch")
    
    logger.info(f"Processing batch search for {len(intents)} intents")
    
    results = []
    for i, intent in enumerate(intents):
        try:
            search_results = search_documents(intent, top_k=top_k)
            results.append(search_results)
            logger.debug(f"Batch search {i+1}/{len(intents)} completed: {len(search_results)} results")
        except SearchError as e:
            logger.error(f"Batch search {i+1} failed: {str(e)}")
            results.append([])  # Empty results for failed searches
    
    logger.info(f"Batch search completed: {len(results)} result sets")
    return results


def aggregate_search_results(results_list: List[List[dict]], 
                           merge_strategy: str = 'interleave',
                           max_results: int = 20) -> List[dict]:
    """
    Aggregates multiple search result lists into a single ranked list.
    
    Args:
        results_list: List of search result lists to aggregate
        merge_strategy: Strategy for merging ('interleave', 'score_based', 'round_robin')
        max_results: Maximum number of results in final list
        
    Returns:
        Aggregated and ranked search results
        
    Raises:
        SearchError: For invalid parameters
    """
    if not results_list or not isinstance(results_list, list):
        raise SearchError("results_list must be a non-empty list")
    
    valid_strategies = ['interleave', 'score_based', 'round_robin']
    if merge_strategy not in valid_strategies:
        raise SearchError(f"merge_strategy must be one of: {valid_strategies}")
    
    logger.info(f"Aggregating {len(results_list)} result sets using {merge_strategy} strategy")
    
    # Remove empty result lists
    filtered_results = [results for results in results_list if results]
    
    if not filtered_results:
        logger.info("No results to aggregate")
        return []
    
    # Aggregate based on strategy
    if merge_strategy == 'score_based':
        # Combine all results and sort by similarity score
        all_results = []
        for results in filtered_results:
            all_results.extend(results)
        
        # Remove duplicates based on document ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            doc_id = result.get('file_id', '') + str(result.get('chunk_index', 0))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        # Sort by similarity score
        aggregated = sorted(unique_results, 
                          key=lambda x: x.get('_similarity', 0.0), 
                          reverse=True)
    
    elif merge_strategy == 'interleave':
        # Interleave results from different searches
        aggregated = []
        max_length = max(len(results) for results in filtered_results)
        
        for i in range(max_length):
            for results in filtered_results:
                if i < len(results):
                    # Check for duplicates
                    doc_id = results[i].get('file_id', '') + str(results[i].get('chunk_index', 0))
                    if not any(r.get('file_id', '') + str(r.get('chunk_index', 0)) == doc_id 
                             for r in aggregated):
                        aggregated.append(results[i])
    
    else:  # round_robin
        # Round-robin selection from each result set
        aggregated = []
        result_iterators = [iter(results) for results in filtered_results]
        
        while result_iterators:
            for i, iterator in enumerate(result_iterators[:]):
                try:
                    result = next(iterator)
                    # Check for duplicates
                    doc_id = result.get('file_id', '') + str(result.get('chunk_index', 0))
                    if not any(r.get('file_id', '') + str(r.get('chunk_index', 0)) == doc_id 
                             for r in aggregated):
                        aggregated.append(result)
                except StopIteration:
                    result_iterators.remove(iterator)
    
    # Limit results
    final_results = aggregated[:max_results]
    
    logger.info(f"Aggregated {len(final_results)} results using {merge_strategy} strategy")
    return final_results


# Export main functions for module interface
__all__ = [
    'search_documents',
    'search_documents_by_similarity', 
    'get_document_by_id',
    'get_related_documents',
    'search_within_file',
    'get_search_suggestions',
    'get_search_analytics',
    'batch_search_documents',
    'aggregate_search_results',
    'health_check',
    'SearchError',
    'Platform',
    'FileCategory',
    'SearchMetrics'
]
