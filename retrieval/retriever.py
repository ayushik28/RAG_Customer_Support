from typing import Optional, Dict, Any, List, Tuple
from langchain_core.documents import Document

# --------------------------------------------------
# Core Retrieval Function (with optional filtering)
# --------------------------------------------------

def retrieve_with_scores(
    query: str,
    vectorstore,
    doc_type: Optional[str] = None,
    k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k documents with distance scores.
    Lower score = higher semantic similarity (Chroma distance).
    """

    if doc_type:
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter={"document_type": doc_type}
        )
        retrieval_mode = "FILTERED"
    else:
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        retrieval_mode = "GLOBAL"

    return results, retrieval_mode

# --------------------------------------------------
# Retrieval with Relevance Guardrail
# --------------------------------------------------

def retrieve_with_relevance_guardrail(
    query: str,
    vectorstore,
    doc_type: Optional[str] = None,
    k: int = 5,
    score_threshold: float = 1.5,
    min_docs: int = 1
) -> Dict[str, Any]:
    """
    Applies strict relevance filtering and returns
    a structured retrieval response for downstream LangGraph nodes.
    """

    results, retrieval_mode = retrieve_with_scores(
        query=query,
        vectorstore=vectorstore,
        doc_type=doc_type,
        k=k
    )

    # Chroma returns distance scores (lower = better)
    relevant_docs = [
        (doc, score)
        for doc, score in results
        if score <= score_threshold
    ]

    # Sort by best relevance
    relevant_docs = sorted(relevant_docs, key=lambda x: x[1])

    # Guardrail: No or insufficient relevant documents
    if len(relevant_docs) < min_docs:
        return {
            "status": "NO_RELEVANT_DOCS",
            "message": "The requested information is not available in the internal documentation.",
            "retrieval_mode": retrieval_mode,
            "query": query,
            "results": []
        }

    return {
        "status": "SUCCESS",
        "message": "Relevant documents retrieved successfully.",
        "retrieval_mode": retrieval_mode,
        "query": query,
        "results": relevant_docs
    }
