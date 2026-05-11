import time
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import logger

class LLMService:
    def __init__(self):
        model_name = "openai/gpt-4o-mini" if settings.OPENAI_BASE_URL else "gpt-4o-mini"
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a highly accurate RAG assistant. Use the provided context to answer the user's question.
        
        Rules:
        1. Answer ONLY from the context.
        2. If the context does not contain the answer, say "I don't have enough information to answer this based on the provided documents."
        3. Be concise and professional.
        4. Cite your sources by referring to the filename and page/section if available.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        start_time = time.time()
        
        context_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}]\n{doc.page_content}" 
            for doc in context_docs
        ])
        
        response = self.chain.invoke({
            "context": context_text,
            "question": query
        })
        
        latency = time.time() - start_time
        logger.info("generation_finished", latency=latency)
        
        return response, latency
