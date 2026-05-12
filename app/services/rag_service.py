import time
from typing import List, Dict, Any, Optional
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.reranking import RerankingService
from app.services.llm import LLMService
from app.core.logging import logger

class RAGService:
    def __init__(self):
        self.ingestion_service = IngestionService()
        self.retrieval_service = RetrievalService()
        self.reranking_service = RerankingService()
        self.llm_service = LLMService()

    async def query(self, query_text: str, filter: Optional[dict] = None) -> Dict[str, Any]:
        overall_start = time.time()
        
        # 1. grab initial stuff
        initial_docs, retrieval_latency = self.retrieval_service.search(query_text, filter=filter)
        
        # 2. rerank them so the best ones are at the top
        reranked_docs, reranking_latency = self.reranking_service.rerank(query_text, initial_docs)
        
        # 3. let the model spit out the final answer
        answer, generation_latency = self.llm_service.generate_response(query_text, reranked_docs)
        
        overall_latency = time.time() - overall_start
        
        logger.info("query_pipeline_finished", total_latency=overall_latency)
        
        return {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in reranked_docs
            ],
            "metrics": {
                "retrieval_latency": retrieval_latency,
                "reranking_latency": reranking_latency,
                "generation_latency": generation_latency,
                "total_latency": overall_latency
            }
        }

    async def ingest_file(self, file_path: str, category: str = "general"):
        chunks = self.ingestion_service.process_document(file_path, category)
        self.retrieval_service.add_documents(chunks)
        return len(chunks)
