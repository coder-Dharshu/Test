"""
Centralized Configuration & Validation Module
=============================================
Reads and validates environment variables once, avoiding drift across services.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

class AppConfig(BaseModel):
    # Inference configuration
    INFERENCE_BACKEND: Literal["groq", "local", "openai_compatible"] = Field(
        default_factory=lambda: os.getenv("INFERENCE_BACKEND", "groq").lower()  # type: ignore
    )
    ALLOW_CLOUD_ESCALATION: bool = Field(
        default_factory=lambda: os.getenv("ALLOW_CLOUD_ESCALATION", "true").lower() in ("true", "1", "yes")
    )
    GROQ_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or os.getenv("QWEN_API_KEY", "")
    )
    QWEN_MODEL: str = Field(
        default_factory=lambda: os.getenv("QWEN_MODEL") or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    )
    LOCAL_MODEL_URL: str = Field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_URL") or os.getenv("QWEN_LOCAL_URL", "http://localhost:11434/v1")
    )
    LOCAL_MODEL_NAME: str = Field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_NAME", "qwen3.6:27b")
    )
    LOCAL_API_KEY: str = Field(
        default_factory=lambda: os.getenv("LOCAL_API_KEY", "local-qwen")
    )
    OPENAI_COMPATIBLE_URL: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_URL", "")
    )
    OPENAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # Generation parameters
    TEMPERATURE: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    MAX_TOKENS: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS") or os.getenv("QWEN_MAX_TOKENS", "4096"))
    )
    RESPONSE_FORMAT: str = Field(
        default_factory=lambda: os.getenv("RESPONSE_FORMAT", "json_object")
    )
    VLM_HARD_TIMEOUT: int = Field(
        default_factory=lambda: int(os.getenv("VLM_HARD_TIMEOUT", "60"))
    )
    VLM_MAX_RETRIES: int = Field(
        default_factory=lambda: int(os.getenv("VLM_MAX_RETRIES", "3"))
    )

    # Microservice Ports
    PORT_GATEWAY: int = Field(
        default_factory=lambda: int(os.getenv("PORT_GATEWAY", "8000"))
    )
    PORT_TRIAGE: int = Field(
        default_factory=lambda: int(os.getenv("PORT_TRIAGE", "8001"))
    )
    PORT_QWEN: int = Field(
        default_factory=lambda: int(os.getenv("PORT_QWEN", "8002"))
    )
    PORT_HITL_GRADIO: int = Field(
        default_factory=lambda: int(os.getenv("PORT_HITL_GRADIO", "7860"))
    )
    PORT_HITL_STREAMLIT: int = Field(
        default_factory=lambda: int(os.getenv("PORT_HITL_STREAMLIT", "8501"))
    )
    PORT_ANALYTICS: int = Field(
        default_factory=lambda: int(os.getenv("PORT_ANALYTICS", "8003"))
    )

    # Internal Service URLs
    LEGO2_URL: str = Field(
        default_factory=lambda: os.getenv("LEGO2_URL", "http://localhost:8001")
    )
    LEGO3_URL: str = Field(
        default_factory=lambda: os.getenv("LEGO3_URL", "http://localhost:8002")
    )
    GATEWAY_URL: str = Field(
        default_factory=lambda: os.getenv("GATEWAY_URL", "http://localhost:8000")
    )
    ANALYTICS_URL: str = Field(
        default_factory=lambda: os.getenv("ANALYTICS_URL", "http://localhost:8003")
    )

    # Storage paths
    UPLOAD_DIR: str = Field(
        default_factory=lambda: os.getenv("UPLOAD_DIR", "./temp_uploads")
    )
    PENDING_DIR: str = Field(
        default_factory=lambda: os.getenv("PENDING_DIR", "./pending_review")
    )
    FINAL_DIR: str = Field(
        default_factory=lambda: os.getenv("FINAL_DIR", "./final_database")
    )
    REJECTED_DIR: str = Field(
        default_factory=lambda: os.getenv("REJECTED_DIR", "./rejected")
    )
    LEGO2_TEMP: str = Field(
        default_factory=lambda: os.getenv("LEGO2_TEMP", "./lego2_temp")
    )
    ASSETS_DIR: str = Field(
        default_factory=lambda: os.getenv("ASSETS_DIR", "./extracted_assets")
    )
    EXPORT_DIR: str = Field(
        default_factory=lambda: os.getenv("EXPORT_DIR", "./exports")
    )
    AUDIT_DIR: str = Field(
        default_factory=lambda: os.getenv("AUDIT_DIR", "./audit")
    )

    # Security
    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "greencare-dev-secret-change-in-production")
    )
    WEBHOOK_SECRET: str = Field(
        default_factory=lambda: os.getenv("WEBHOOK_SECRET", "greencare-webhook-secret-change-in-production")
    )
    ANALYTICS_API_KEY: str = Field(
        default_factory=lambda: os.getenv("ANALYTICS_API_KEY", "")
    )
    HITL_ADMIN_PASSWORD: str = Field(
        default_factory=lambda: os.getenv("HITL_ADMIN_PASSWORD", "")
    )

    # Ingestion constraints
    MAX_FILE_SIZE_MB: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    )
    MAX_PAGES_PER_DOC: int = Field(
        default_factory=lambda: int(os.getenv("MAX_PAGES_PER_DOC", "100"))
    )
    MAX_BATCH_FILES: int = Field(
        default_factory=lambda: int(os.getenv("MAX_BATCH_FILES", "20"))
    )
    PER_FILE_TIMEOUT_SECONDS: int = Field(
        default_factory=lambda: int(os.getenv("PER_FILE_TIMEOUT_SECONDS", "120"))
    )

    # Smart triage thresholds
    PCS_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("PCS_THRESHOLD", "0.50"))
    )
    OCR_CONFIDENCE_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "80.0"))
    )

    def validate_inference_backend(self) -> None:
        """Fail fast if inference backend is improperly configured."""
        if self.INFERENCE_BACKEND == "groq":
            if not self.GROQ_API_KEY or self.GROQ_API_KEY.strip() in ("", "your_groq_api_key_here", "your_api_key_here"):
                raise ValueError(
                    "Config Error: INFERENCE_BACKEND is set to 'groq', but GROQ_API_KEY is missing or unconfigured. "
                    "Set GROQ_API_KEY in .env or switch to INFERENCE_BACKEND=local."
                )
        elif self.INFERENCE_BACKEND == "local":
            if not self.LOCAL_MODEL_URL or not self.LOCAL_MODEL_URL.startswith("http"):
                raise ValueError(
                    "Config Error: INFERENCE_BACKEND is set to 'local', but LOCAL_MODEL_URL is invalid or missing. "
                    "Expected e.g. 'http://localhost:11434/v1'."
                )
        elif self.INFERENCE_BACKEND == "openai_compatible":
            if not self.OPENAI_API_KEY and not self.GROQ_API_KEY:
                raise ValueError(
                    "Config Error: INFERENCE_BACKEND is set to 'openai_compatible', but no API key is provided."
                )

    def ensure_directories(self) -> None:
        """Create all required persistent directories."""
        for d in [
            self.UPLOAD_DIR, self.PENDING_DIR, self.FINAL_DIR,
            self.REJECTED_DIR, self.LEGO2_TEMP, self.ASSETS_DIR,
            self.EXPORT_DIR, self.AUDIT_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


# Global singleton configuration
config = AppConfig()
