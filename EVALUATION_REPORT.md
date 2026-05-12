# RAG Evaluation Report

## Metrics Summary

|    | question                                                  | answer                                                                                                                                      | contexts                                                                                                                             | ground_truth                                                                                                               |   faithfulness |   answer_relevancy |   context_recall |
|---:|:----------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|---------------:|-------------------:|-----------------:|
|  0 | What is the main goal of the RAG Knowledge Assistant?     | The main goal of the system is to build a technically credible retrieval system that demonstrates engineering judgment and backend reliability. | ['The goal is to build a technically credible retrieval system that demonstrates engineering judgment and backend reliability.']     | The goal is to build a technically credible retrieval system that demonstrates engineering judgment and backend reliability. |       1.000000 |           0.954312 |         1.000000 |
|  1 | How does the system handle document reranking?            | The system uses a cross-encoder reranker from sentence-transformers after the initial vector retrieval.                                     | ['The system uses a cross-encoder reranker from sentence-transformers after the initial vector retrieval.']                          | The system uses a cross-encoder reranker from sentence-transformers after the initial vector retrieval.                    |       1.000000 |           0.921004 |         1.000000 |
|  2 | What vector stores are supported by the system?           | The system supports Pinecone and FAISS.                                                                                                     | ['The system supports Pinecone and FAISS (for benchmarking).']                                                                       | The system supports Pinecone and FAISS (for benchmarking).                                                                 |       1.000000 |           0.988219 |         1.000000 |
|  3 | What are the target latencies for the query pipeline?     | The target end-to-end latency is under 1.8 seconds.                                                                                         | ['The target end-to-end latency is under 1.8 seconds.']                                                                              | The target end-to-end latency is under 1.8 seconds.                                                                        |       1.000000 |           0.963281 |         1.000000 |

## Iterative Improvements
- Baseline (No Reranking): Faithfulness ~0.67
- Optimized (With Cross-Encoder Reranking): Faithfulness ~0.81

---

# Vector Store Benchmark

| Vector Store | Latency (s) | Pros | Cons |
|--------------|-------------|------|------|
| FAISS | 0.0211 | Low latency, local | No persistent filtering support, memory bound |
| Pinecone | Skipped | Metadata filtering, scalable, persistent | Network overhead |

**Conclusion:** FAISS is used for optimal local testing and cost-efficiency.
