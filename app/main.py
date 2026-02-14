from fastapi import FastAPI
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.routes.document_routes import router
from app.models.document import StoredDocument
from app.models.chunk import DocumentChunk
from app.config import settings

app = FastAPI()

@app.on_event("startup")
async def start_db():
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    
    await init_beanie(
        database=client.get_default_database(),
        document_models=[StoredDocument, DocumentChunk]
    )

app.include_router(router)
