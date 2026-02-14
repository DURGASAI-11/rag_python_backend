from beanie import Document
from typing import List
from datetime import datetime
from pydantic import Field

class DocumentChunk(Document):
    document_id: str
    user_id: str
    chunk_index: int
    chunk_text: str
    embedding: List[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "document_chunks"
