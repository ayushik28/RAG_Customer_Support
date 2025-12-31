import json
from pathlib import Path
from langchain_core.documents import Document

from reasoning.graph_state import GraphState
from reasoning.guardrails import redact_pii
from retrieval.retriever import retrieve_with_relevance_guardrail
from ingestion.build_vectorstore import build_vectorstore

# Load vectorstore once (shared across all nodes)
VECTORSTORE = build_vectorstore()

LOG_FILE = Path("logs/audit_logs.jsonl")

def intent_detection_node(state: GraphState) -> GraphState:
    meta_keywords = ["what policies are available", "what documents", "what is available", "list", "overview"]

    if any(k in state.query.lower() for k in meta_keywords):
        state.intent = "OVERVIEW"
    else:
        state.intent = "FACT_LOOKUP"

    return state

def retrieval_node(state: GraphState) -> GraphState:
    if state.intent == "OVERVIEW":
        state.retrieved_docs = []
        return state

    # 🔑 HIGH-RECALL retrieval (do NOT filter yet)
    retrieved = vectorstore.similarity_search(
        state.query,
        k=8  # higher recall so correct FAQ is included
    )

    if not retrieved:
        state.retrieved_docs = []
        state.grounded = False
        state.answer = "I could not find relevant information in the internal documents."
        return state

    state.retrieved_docs = retrieved
    return state


def grounding_node(state: GraphState) -> GraphState:

    if state.intent == "OVERVIEW":
        state.grounded = True
        return state

    if not state.retrieved_docs:
        state.grounded = False
        state.answer = "I could not find relevant information in the internal documents."
        return state

    query_terms = set(state.query.lower().split())
    grounded_docs = []

    for doc in state.retrieved_docs:
        doc_terms = set(doc.page_content.lower().split())
        if query_terms & doc_terms:
            grounded_docs.append(doc)

    if not grounded_docs:
        state.grounded = False
        state.answer = "I could not find relevant information in the internal documents."
        return state

    state.retrieved_docs = grounded_docs
    state.grounded = True
    return state

def safety_guardrail_node(state: GraphState) -> GraphState:

    if state.intent == "OVERVIEW":
        state.safe = True
        return state

    if not state.grounded:
        state.safe = False
        return state

    sanitized_docs = []
    for doc in state.retrieved_docs:
        sanitized_docs.append(
            Document(
                page_content=redact_pii(doc.page_content),
                metadata=doc.metadata
            )
        )

    state.retrieved_docs = sanitized_docs
    state.safe = True
    return state


def response_node(state: GraphState) -> GraphState:
    # 1. Handle overview intent (system-generated)
    if state.intent == "OVERVIEW":
        state.answer = (
            "[System Overview]\n"
            "The following internal documents are available:\n"
            "- FAQs\n"
            "- Service Level Agreements (SLA)\n"
            "- Internal Policies\n"
            "- Escalation Workflows\n\n"
            "Please ask a specific question about any of these documents."
        )
        return state

    # 2. Enforce grounding & safety
    if not state.grounded or not state.safe:
        state.answer = "I could not find relevant information in the internal documents."
        return state

    # 3. Enforce retrieval
    if not state.retrieved_docs:
        state.answer = "I could not find relevant information in the internal documents."
        return state

    # 4. Select primary and supporting documents (fallback only)
    primary_doc = state.retrieved_docs[0]
    supporting_docs = state.retrieved_docs[1:3]
    primary_content = primary_doc.page_content.strip()

    # 5. FAQ-style extraction (semantic + confidence-based + out-of-scope guardrail)
    faq_docs = [
        d for d in state.retrieved_docs
        if d.metadata.get("document_type") == "FAQ"
    ]

    if faq_docs:
        selected_faqs, top_score = select_relevant_faqs(
            state.query,
            faq_docs,
            embedding_model
        )

        # OUT-OF-SCOPE GUARDRAIL
        if top_score < 0.45:
            state.answer = "I could not find relevant information in the internal documents."
            return state

        faq_answers = [
            doc.page_content.split("A:", 1)[1].strip()
            for doc in selected_faqs
            if "A:" in doc.page_content
        ]

        if faq_answers:
            final_answer = "\n\n".join(faq_answers)

            state.answer = (
                f"{final_answer}\n\n"
                f"Primary Source: {selected_faqs[0].metadata.get('source')} (FAQ)"
            )
            return state

    # 6. Fallback for SLA / Policy / Workflow chunks
    state.answer = (
        f"{primary_content}\n\n"
        f"Primary Source: {primary_doc.metadata.get('source')} "
        f"({primary_doc.metadata.get('document_type')})"
    )

    if supporting_docs:
        state.answer += "\nSupporting Sources: " + ", ".join(
            f"{doc.metadata.get('source')} ({doc.metadata.get('document_type')})"
            for doc in supporting_docs
        )

    return state

def log_interaction(state: GraphState):
    log_entry = {
        # Session metadata
        "session_id": state.session_id,
        "timestamp": state.timestamp,

        # User input
        "query": state.query,
        "intent": state.intent,

        # System decision flags
        "grounded": state.grounded,
        "safe": state.safe,
        "status": (
            "SUCCESS" if state.grounded and state.safe
            else "NO_RELEVANT_DOCS"
        ),

        # Final output
        "answer": state.answer,

        # Primary grounding reference (important for audits)
        "primary_chunk_id": (
            state.retrieved_docs[0].metadata.get("chunk_id")
            if state.retrieved_docs else None
        ),

        # Retrieved context (redacted + traceable)
        "retrieved_documents": [
            {
                "source": doc.metadata.get("source"),
                "document_type": doc.metadata.get("document_type"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content_preview": doc.page_content[:200].replace("\n", " ")
            }
            for doc in state.retrieved_docs
        ]
    }

    with LOG_FILE.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")

def audit_log_node(state: GraphState) -> GraphState:
    log_interaction(state)
    return state
