"""
Logging middleware for Energy Optimization API.

This module provides middleware for structured request/response logging.
"""

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.

    Logs all requests with timing, status codes, and error details.

    Examples
    --------
    >>> from fastapi import FastAPI
    >>> from src.api.middleware.logging_middleware import LoggingMiddleware
    >>> app = FastAPI()
    >>> app.add_middleware(LoggingMiddleware)
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process request and log details.

        Parameters
        ----------
        request : Request
            Incoming HTTP request
        call_next : callable
            Next middleware/endpoint

        Returns
        -------
        Response
            HTTP response with timing headers
        """
        # Start timer
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            },
        )

        # Process request
        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000  # ms

            # Log successful response
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time, 2),
                },
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(round(process_time, 2))
            return response

        except Exception as e:
            process_time = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time_ms": round(process_time, 2),
                },
                exc_info=True,
            )
            raise
