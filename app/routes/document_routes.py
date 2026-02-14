from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Depends
from typing import Optional
from pypdf import PdfReader
from app.models.chunk import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.qa_service import QAService
from app.utils.text_utils import clean_text, chunk_text
from app.core.security import verify_access_token  # 🔥 JWT middleware
import numpy as np

router = APIRouter()

embedding_service = EmbeddingService()
qa_service = QAService()


# =============================
# Upload Document (Secure)
# =============================
@router.post("/upload")
async def upload_document(
    document_id: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(verify_access_token)  # 🔥 JWT verification
):

    user_id = user["user_id"]  # Extract from verified token

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found")

    embeddings = embedding_service.embed_texts(chunks)

    chunk_docs = []

    for idx, chunk in enumerate(chunks):
        chunk_docs.append(
            DocumentChunk(
                document_id=document_id,
                user_id=user_id,   # 🔥 From JWT only
                chunk_index=idx,
                chunk_text=chunk,
                embedding=embeddings[idx]
            )
        )

    await DocumentChunk.insert_many(chunk_docs)

    return {
        "message": "Document chunks indexed securely",
        "chunks_created": len(chunks),
        "document_id": document_id
    }


# =============================
# Ask Question (Secure)
# =============================
@router.post("/ask")
async def ask_question(
    question: str,
    document_id: Optional[str] = None,   # 👈 optional now
    current_user=Depends(verify_access_token)  # 👈 your JWT verification dependency
):
    user_id = current_user["user_id"]   # extracted from verified JWT

    # Generate embedding for query
    query_embedding = embedding_service.embed_query(question)

    # Build dynamic query
    if document_id:
        # 🔎 Search specific document
        chunks = await DocumentChunk.find(
            DocumentChunk.user_id == user_id,
            DocumentChunk.document_id == document_id
        ).to_list()
    else:
        # 🔎 Search across ALL documents of that user
        chunks = await DocumentChunk.find(
            DocumentChunk.user_id == user_id
        ).to_list()

    if not chunks:
        raise HTTPException(status_code=404, detail="No documents found")

    # Similarity scoring
    scores = [
        np.dot(query_embedding, chunk.embedding)
        for chunk in chunks
    ]

    # Top 5 relevant chunks
    top_indices = np.argsort(scores)[-5:][::-1]
    top_chunks = [chunks[i].chunk_text for i in top_indices]

    context = " ".join(top_chunks)

    answer = qa_service.generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "searched_document": document_id if document_id else "ALL_USER_DOCUMENTS"
    }