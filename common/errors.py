"""
Standardized Error Taxonomy for Greencare AI / Pramaan IDP
==========================================================
Every API error returns {error_code, message, retryable}.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class ErrorCode(str, Enum):
    # Ingestion & Validation
    UNSUPPORTED_MIME    = "UNSUPPORTED_MIME"
    FILE_TOO_LARGE      = "FILE_TOO_LARGE"
    BATCH_LIMIT         = "BATCH_LIMIT"
    MAGIC_BYTES_MISMATCH = "MAGIC_BYTES_MISMATCH"
    PAGE_LIMIT_EXCEEDED = "PAGE_LIMIT_EXCEEDED"
    CORRUPTED_FILE      = "CORRUPTED_FILE"

    # Extraction & Triage
    OCR_FAILED          = "OCR_FAILED"
    TRIAGE_FAILED       = "TRIAGE_FAILED"
    VLM_TIMEOUT         = "VLM_TIMEOUT"
    VLM_RATE_LIMITED    = "VLM_RATE_LIMITED"
    VLM_UNAVAILABLE     = "VLM_UNAVAILABLE"
    CLOUD_DISABLED      = "CLOUD_DISABLED"
    SCHEMA_INVALID      = "SCHEMA_INVALID"

    # Routing & Jobs
    JOB_NOT_FOUND       = "JOB_NOT_FOUND"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    UNAUTHORIZED        = "UNAUTHORIZED"
    INTERNAL_ERROR      = "INTERNAL_ERROR"


class APIErrorDetail(BaseModel):
    error_code: ErrorCode
    message: str
    retryable: bool
    details: Optional[dict] = None


class IDPException(HTTPException):
    """Custom HTTP Exception carrying structured APIErrorDetail."""
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        retryable: bool = False,
        details: Optional[dict] = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = error_code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> dict:
        data = {
            "error_code": self.error_code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            data["details"] = self.details
        return data

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.to_dict())
