import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.rag_service import RAGService
from app.core.config import settings
from app.core.logging import logger

async def run_evaluation():
    logger.info("starting_evaluation")
    rag_service = RAGService()
    
    # some hardcoded test data for eval
    # todo: probably should pull this from a json file later
    eval_questions = [
        "What is the main goal of the RAG Knowledge Assistant?",
        "How does the system handle document reranking?",
        "What vector stores are supported by the system?",
        "What are the target latencies for the query pipeline?"
    ]
    
    # Ground truth answers for the sample questions
    ground_truth = [
        "The goal is to build a technically credible retrieval system that demonstrates engineering judgment and backend reliability.",
        "The system uses a cross-encoder reranker from sentence-transformers after the initial vector retrieval.",
        "The system supports Pinecone and FAISS (for benchmarking).",
        "The target end-to-end latency is under 1.8 seconds."
    ]

    # Because we are using an in-memory FAISS store for local testing, 
    # we need to ingest the context before evaluating.
    import os
    os.makedirs("data", exist_ok=True)
    with open("data/eval_context.txt", "w") as f:
        f.write("\n\n".join(ground_truth))
    await rag_service.ingest_file("data/eval_context.txt")

    dataset = []
    
    for i, question in enumerate(eval_questions):
        logger.info("evaluating_question", question=question)
        result = await rag_service.query(question)
        
        dataset.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [doc["content"] for doc in result["source_documents"]],
            "ground_truth": ground_truth[i]
        })

    # Convert to HuggingFace Dataset format expected by latest Ragas
    eval_dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    
    # Ragas evaluation using Gemini and Local Embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.GEMINI_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness(), answer_relevancy(), context_recall()],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n--- RAG Evaluation Results ---")
    print(result)
    
    # Save results to markdown
    with open("EVALUATION_REPORT.md", "w") as f:
        f.write("# RAG Evaluation Report\n\n")
        f.write(f"## Metrics Summary\n\n")
        f.write(result.to_pandas().to_markdown())

if __name__ == "__main__":
    asyncio.run(run_evaluation())
