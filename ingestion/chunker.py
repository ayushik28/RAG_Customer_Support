import re

def chunk_faq(text: str):
    blocks = re.split(r"(?=Q:\s)", text)
    return [b.strip() for b in blocks if "A:" in b]

def chunk_numbered_sections(text: str):
    text = text.replace("\r", " ").strip()

    sections = re.split(
        r"(?=\b\d+\.\d+\s+[A-Z])",
        text
    )

    return [
        s.strip()
        for s in sections
        if len(s.strip()) > 120
    ]

from langchain_core.documents import Document

def chunk_by_structure(doc):
    text = doc["page_content"]
    doc_type = doc["metadata"]["document_type"]

    if doc_type == "FAQ":
        return chunk_faq(text)

    return chunk_numbered_sections(text)
