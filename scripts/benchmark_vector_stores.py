import time
from app.services.retrieval import RetrievalService
from langchain_core.documents import Document
from app.core.logging import logger

def benchmark_vector_stores():
    sample_docs = [
        Document(page_content="RAG is a technique to improve LLM accuracy.", metadata={"source": "doc1"}),
        Document(page_content="Vector databases like Pinecone are essential for RAG.", metadata={"source": "doc2"}),
        Document(page_content="FAISS is a fast local library for vector similarity search.", metadata={"source": "doc3"}),
    ] * 10 # multiply by 10 to give the db somthing to chew on

    # Test FAISS
    logger.info("benchmarking_faiss")
    faiss_service = RetrievalService(use_faiss=True)
    faiss_service.add_documents(sample_docs)
    
    start = time.time()
    _, _ = faiss_service.search("What is RAG?", top_k=5)
    faiss_latency = time.time() - start
    
    print("\n--- Vector Store Benchmark ---")
    print(f"FAISS Retrieval Latency: {faiss_latency:.4f}s")
    print(f"(Pinecone benchmarking skipped in local configuration)")
    
    with open("BENCHMARK_RESULTS.md", "w") as f:
        f.write("# Vector Store Benchmark\n\n")
        f.write("| Vector Store | Latency (s) | Pros | Cons |\n")
        f.write("|--------------|-------------|------|------|\n")
        f.write(f"| FAISS | {faiss_latency:.4f} | Low latency, local | No persistent filtering support, memory bound |\n")
        f.write("| Pinecone | Skipped | Metadata filtering, scalable, persistent | Network overhead |\n")

if __name__ == "__main__":
    benchmark_vector_stores()
