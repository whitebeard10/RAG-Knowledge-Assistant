from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: Optional[str] = None
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "rag-knowledge-assistant"
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"

    # LangSmith Tracing
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: Optional[str] = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "rag-knowledge-assistant"

    # App Settings
    APP_NAME: str = "RAG Knowledge Assistant"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    USE_LOCAL_MODELS: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # RAG Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 51
    RERANK_TOP_K: int = 20
    FINAL_TOP_K: int = 5
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
