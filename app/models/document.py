from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import Optional

class StoredDocument(Document):
    user_id: str
    file_name: str
    total_pages: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "documents"
