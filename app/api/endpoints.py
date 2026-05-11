import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.schemas.request_models import QueryRequest, QueryResponse, IngestResponse, HealthResponse
from app.services.rag_service import RAGService
from app.core.logging import logger

router = APIRouter()
rag_service = RAGService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = await rag_service.query(request.query, filter=request.filter)
        return result
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error during query processing")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...), 
    category: str = "general"
):
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    temp_path = f"data/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        chunks_count = await rag_service.ingest_file(temp_path, category)
        
        # In a real production app, we might want to delete the temp file after processing
        # os.remove(temp_path)
        
        return {
            "message": "Document ingested successfully",
            "chunks_count": chunks_count,
            "file_name": file.filename
        }
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")
