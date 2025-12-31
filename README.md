# Customer Support Intelligence System

This project implements a Retrieval-Augmented Generation (RAG) based
Customer Support Intelligence System using LangGraph orchestration
and strict grounding and safety guardrails.

## Features
- Semantic retrieval using Chroma
- LangGraph-based reasoning
- Grounding enforcement
- PII redaction
- Audit logging
- Gradio UI

## How to Run
pip install -r requirements.txt  
python ingestion/build_vectorstore.py  
python ui/app.py


# Customer Support Intelligence System  
### RAG-based System with LangGraph Orchestration & Guardrails

---

## Project Overview

Customer support teams rely on internal documentation such as FAQs, Service Level Agreements (SLAs), internal policies, and escalation workflows to answer user queries accurately. Traditional chatbot systems often hallucinate responses or generate unverified information, leading to compliance risks and poor user experience.

This project implements a **Customer Support Intelligence System** using **Retrieval-Augmented Generation (RAG)** principles combined with **LangGraph-based reasoning and safety guardrails**. The system answers customer queries **strictly using internal documentation**, enforces grounding, prevents hallucinations, and maintains full auditability.

---

## Key Objectives

- Retrieve answers only from verified internal documents  
- Prevent hallucinations and unsafe responses  
- Enforce grounding before response generation  
- Redact sensitive information (PII)  
- Maintain session-level audit logs  
- Provide an explainable and auditable customer support interface  

---

## System Architecture (High-Level)

User Query
↓
Intent Detection (LangGraph)
↓
Semantic Retrieval (Vector DB)
↓
Grounding Validation
↓
Safety & PII Guardrails
↓
Deterministic Response Synthesis
↓
Audit Logging
↓
Gradio UI Response



---

## Project Structure

customer-support-rag-langgraph/
│
├── data/
│ ├── raw/ # Original PDF documents
│ └── processed/
│
├── ingestion/ # Task 1: Document ingestion & indexing
│ ├── pdf_loader.py
│ ├── cleaner.py
│ ├── chunker.py
│ └── build_vectorstore.py
│
├── retrieval/ # Task 2: Semantic retrieval
│ └── retriever.py
│
├── reasoning/ # Tasks 3–5: LangGraph reasoning & guardrails
│ ├── graph_state.py
│ ├── guardrails.py
│ ├── nodes.py
│ └── graph.py
│
├── ui/ # Task 6: Gradio UI
│ └── app.py
│
├── logs/ # Audit logs (JSONL)
│ └── audit_logs.jsonl
│
├── notebooks/
│ └── final_execution.ipynb
│
├── chroma_db/ # Persisted vector database
│
├── requirements.txt
└── README.md



---

## Task-wise Implementation Summary

### 🔹 Task 1: Document Processing & Vector Indexing
- PDF ingestion using `pypdf`
- Document-type-aware chunking:
  - FAQs → Q&A chunks  
  - SLAs / Policies → Section-based chunks  
- Metadata enrichment (source, document type, chunk ID)  
- Embeddings generated using `sentence-transformers`  
- Vector storage using **Chroma**

---

### 🔹 Task 2: RAG-Based Semantic Retrieval
- Semantic similarity search over vector database  
- Top-k retrieval with relevance threshold  
- Explicit handling of empty or low-confidence retrievals  

---

### 🔹 Task 3: LangGraph-Based Reasoning
- Deterministic node-based reasoning using LangGraph  
- Modular nodes:
  - Intent detection  
  - Retrieval  
  - Grounding validation  
  - Safety checks  
  - Response synthesis  
  - Audit logging  
- State passed explicitly across nodes  

---

### 🔹 Task 4: Grounding Enforcement & Safety Controls
- Responses generated **only from retrieved content**  
- No generative LLM used to eliminate hallucination risk  
- Semantic confidence thresholds enforced  
- PII redaction applied before response synthesis  
- Explicit refusal for unsupported queries  

> Although the task description references an LLM, this system intentionally avoids generative models to ensure deterministic, grounded, and auditable responses.

---

### 🔹 Task 5: Session Management & Audit Logging
- Each query logged with:
  - Session ID  
  - Timestamp  
  - User query  
  - Retrieved documents  
  - Final answer  
- Logs stored in JSONL format for audit and debugging  

---

### 🔹 Task 6: End-to-End System Integration
- Gradio-based conversational UI  
- Displays:
  - Grounded answer  
  - Source document citations  
- Consistent behavior across sessions  
- Tested with:
  - FAQ queries  
  - SLA / Policy queries  
  - Compound queries  
  - Out-of-scope queries  
