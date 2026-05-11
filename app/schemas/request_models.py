from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the key architectural components of this system?")
    filter: Optional[Dict[str, Any]] = Field(None, example={"category": "architecture"})

class DocumentMetadata(BaseModel):
    source: str
    page: int
    category: str
    ingestion_timestamp: str

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class LatencyMetrics(BaseModel):
    retrieval_latency: float
    reranking_latency: float
    generation_latency: float
    total_latency: float

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
    metrics: LatencyMetrics

class IngestResponse(BaseModel):
    message: str
    chunks_count: int
    file_name: str

class HealthResponse(BaseModel):
    status: str
    version: str
