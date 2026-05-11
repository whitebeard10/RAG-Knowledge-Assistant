import datetime
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import logger

class IngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    def load_document(self, file_path: str) -> List[Document]:
        logger.info("loading_document", file_path=file_path)
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        return loader.load()

    def process_document(self, file_path: str, category: str = "general") -> List[Document]:
        documents = self.load_document(file_path)
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source": file_path.split("/")[-1],
                "category": category,
                "ingestion_timestamp": datetime.datetime.now().isoformat(),
                "page": doc.metadata.get("page", 0)
            })

        chunks = self.text_splitter.split_documents(documents)
        logger.info("document_processed", chunks_count=len(chunks), file_path=file_path)
        return chunks
