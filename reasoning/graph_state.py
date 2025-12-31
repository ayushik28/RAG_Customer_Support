from dataclasses import dataclass, field
from typing import List, Optional
from langchain_core.documents import Document
from datetime import datetime
import uuid

@dataclass
class GraphState:
    # Core
    query: str

    # Session
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Reasoning
    intent: Optional[str] = None
    retrieved_docs: List[Document] = field(default_factory=list)
    grounded: bool = False
    safe: bool = True

    # Output
    answer: Optional[str] = None
