import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger

class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            "request_processed",
            path=request.url.path,
            method=request.method,
            process_time=process_time,
            status_code=response.status_code
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
