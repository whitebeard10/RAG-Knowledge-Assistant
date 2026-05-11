from fastapi import FastAPI
from app.api.endpoints import router
from app.api.middleware import RequestTimingMiddleware
from app.core.logging import setup_logging
from app.core.config import settings

setup_logging()

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade RAG Knowledge Assistant with Cross-Encoder Reranking",
    version="1.0.0"
)

# Add Middleware
app.add_middleware(RequestTimingMiddleware)

# Include Routers
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
