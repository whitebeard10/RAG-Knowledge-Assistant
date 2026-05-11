import time
from typing import List, Optional
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from app.core.config import settings
from app.core.logging import logger

class RetrievalService:
    def __init__(self, use_faiss: bool = None):
        if settings.USE_LOCAL_MODELS:
            logger.info("using_local_embeddings", model="sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.use_faiss = True # Force FAISS if local
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=settings.EMBEDDING_MODEL,
                openai_api_base=settings.OPENAI_BASE_URL
            )
            self.use_faiss = use_faiss if use_faiss is not None else False
        
        if not self.use_faiss:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
        else:
            self.vector_store = None # Initialized on demand for FAISS

    def search(self, query: str, top_k: int = settings.RERANK_TOP_K, filter: Optional[dict] = None) -> List:
        if self.vector_store is None:
            logger.warning("vector_store_not_initialized", use_faiss=self.use_faiss)
            return [], 0.0

        start_time = time.time()
        logger.info("retrieval_started", query=query, top_k=top_k, filter=filter)
        
        docs = self.vector_store.similarity_search(
            query, 
            k=top_k,
            filter=filter
        )
        
        latency = time.time() - start_time
        logger.info("retrieval_finished", latency=latency, docs_count=len(docs))
        return docs, latency

    def add_documents(self, documents: List):
        if self.use_faiss:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
        else:
            self.vector_store.add_documents(documents)
