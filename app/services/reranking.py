import time
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import logger

class RerankingService:
    def __init__(self):
        logger.info("initializing_cross_encoder", model=settings.CROSS_ENCODER_MODEL)
        self.model = CrossEncoder(settings.CROSS_ENCODER_MODEL)

    def rerank(self, query: str, documents: List[Document], top_n: int = settings.FINAL_TOP_K) -> List[Document]:
        if not documents:
            return [], 0.0

        start_time = time.time()
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # sort by score desending so best matches are first
        reranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Take top N
        final_docs = [doc for doc, score in reranked_docs[:top_n]]
        
        latency = time.time() - start_time
        logger.info("reranking_finished", latency=latency, input_count=len(documents), output_count=len(final_docs))
        
        return final_docs, latency
