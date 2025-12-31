
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_faq, chunk_sections


def build_vectorstore():
    """
    Loads PDFs, chunks documents, creates embeddings,
    and persists a Chroma vector store.
    """

    pdfs = {
        "FAQ": "data/raw/FoodDeliveryApp_FAQs.pdf",
        "SLA": "data/raw/FoodDeliveryApp_SLAs.pdf",
        "POLICY": "data/raw/FoodDeliveryApp_Internal_Policies.pdf",
        "ESCALATION": "data/raw/FoodDeliveryApp_Escalation_Workflow.pdf"
    }

    final_documents = []

    # -------------------------------
    # Load, clean, chunk documents
    # -------------------------------
    for doc_type, path in pdfs.items():
        raw_text = load_pdf(path)
        cleaned_text = clean_text(raw_text)

        # Choose chunking strategy
        if doc_type == "FAQ":
            chunks = chunk_faq(cleaned_text)
        else:
            chunks = chunk_sections(cleaned_text)

        for i, chunk in enumerate(chunks):
            metadata = {
                "document_type": doc_type,
                "source": os.path.basename(path),
                "chunk_id": f"{doc_type}_{i}",
                "chunk_type": "faq" if doc_type == "FAQ" else "section"
            }

            # -------------------------------
            # FAQ handling
            # -------------------------------
            if doc_type == "FAQ":
                # chunk is a STRING: "Q: ... A: ..."
                if "Q:" in chunk and "A:" in chunk:
                    faq_question = chunk.split("Q:", 1)[1].split("A:", 1)[0].strip()
                    metadata["faq_question"] = faq_question

                page_content = chunk  # ✅ always string
            else:
                page_content = chunk  # non-FAQ is also string

            final_documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            )

    # -------------------------------
    # Create embeddings
    # -------------------------------
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------------
    # Build & persist Chroma DB
    # -------------------------------
    vectordb = Chroma.from_documents(
        documents=final_documents,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    vectordb.persist()
    return vectordb
