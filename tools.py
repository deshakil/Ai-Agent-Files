# ai_layer/tools.py

from typing import List, Dict, Any

from ai_layer.embedder import get_query_embedding
from ai_layer.search import search_documents_by_similarity
from ai_layer.summarizer import summarize_document
from ai_layer.rag import answer_query_with_rag

# ===========================
# 🔍 Search Tool Wrapper
# ===========================
def tool_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query_text = args.get("query_text")
    user_id = args.get("user_id")
    top_k = args.get("top_k", 10)

    if not query_text or not user_id:
        return {"error": "Missing required parameters: query_text and user_id"}

    # Optional filters
    filters = {}
    if args.get("platform"):
        filters["platform"] = args["platform"]
    if args.get("mime_type"):
        filters["mime_type"] = args["mime_type"]
    if args.get("time_range"):
        filters["time_range"] = args["time_range"]

    # Generate embedding
    embedding: List[float] = get_query_embedding(query_text)

    # Perform semantic search
    results = search_documents_by_similarity(
        embedding=embedding,
        user_id=user_id,
        filters=filters if filters else None,
        top_k=top_k
    )

    return {
        "tool_used": "search",
        "query_text": query_text,
        "filters_applied": filters,
        "results": results
    }


# ===========================
# 🧠 Summarize Tool Wrapper
# ===========================
def tool_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return summarize_document(args)
    except Exception as e:
        return {"error": str(e)}


# ===========================
# 📖 RAG Tool Wrapper
# ===========================
def tool_rag(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return answer_query_with_rag(args)
    except Exception as e:
        return {"error": str(e)}


# ===========================
# 🛠 Tool Function Registry
# ===========================
TOOL_FUNCTIONS = {
    "search": {
        "function": tool_search,
        "spec": {
            "name": "search",
            "description": "Search relevant document chunks based on semantic similarity and optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "The user's search query"},
                    "user_id": {"type": "string", "description": "The user ID"},
                    "platform": {"type": "string", "description": "Platform like 'Drive', 'Slack', etc."},
                    "mime_type": {"type": "string", "description": "MIME type like 'application/pdf'"},
                    "time_range": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string", "format": "date"},
                            "to": {"type": "string", "format": "date"}
                        },
                        "description": "Time range filter for document creation"
                    },
                    "top_k": {"type": "integer", "description": "Number of top results to return"}
                },
                "required": ["query_text", "user_id"]
            }
        }
    },
    "summarize": {
        "function": tool_summarize,
        "spec": {
            "name": "summarize",
            "description": "Summarize a document by file ID or based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string", "description": "The ID of the file to summarize"},
                    "query_text": {"type": "string", "description": "Optional query-based summarization"},
                    "summary_type": {"type": "string", "description": "short, detailed, bullet, etc."},
                    "user_id": {"type": "string", "description": "The user ID"}
                },
                "required": ["user_id"]
            }
        }
    },
    "rag": {
        "function": tool_rag,
        "spec": {
            "name": "rag",
            "description": "Answer a question using RAG (retrieval-augmented generation) over user documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "The user's question"},
                    "user_id": {"type": "string", "description": "The user ID"}
                },
                "required": ["query_text", "user_id"]
            }
        }
    }
}
