"""
Global error handlers for Energy Optimization API.

This module provides exception handlers for FastAPI application,
ensuring consistent error responses.
"""

import logging
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors with detailed messages.

    Parameters
    ----------
    request : Request
        HTTP request that caused validation error
    exc : RequestValidationError
        Pydantic validation exception

    Returns
    -------
    JSONResponse
        422 response with validation error details

    Examples
    --------
    >>> app.add_exception_handler(RequestValidationError, validation_exception_handler)
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join([str(loc) for loc in error["loc"]])
        message = error["msg"]
        error_type = error["type"]

        errors.append(
            {
                "field": field,
                "message": message,
                "type": error_type,
                "input": error.get("input"),
            }
        )

    logger.warning(f"Validation error on {request.url.path}", extra={"errors": errors})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Request data validation failed",
            "details": errors,
            "path": str(request.url.path),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle uncaught exceptions with generic error response.

    Parameters
    ----------
    request : Request
        HTTP request that caused error
    exc : Exception
        Unhandled exception

    Returns
    -------
    JSONResponse
        500 response with error details

    Examples
    --------
    >>> app.add_exception_handler(Exception, general_exception_handler)
    """
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if logger.level <= logging.DEBUG else None,
            "path": str(request.url.path),
        },
    )
