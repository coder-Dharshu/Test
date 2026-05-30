"""
Smart Triage Enterprise IDP Platform
=====================================
A high-efficiency document processing system implementing:
  - Track A:  Multi-engine Digital PDF detection (pypdf + pdfplumber + PyMuPDF)
              with printable-character-ratio gate → zero GPU/API cost
  - Track B / Path 1: CPU OCR via pytesseract (OCR confidence ≥ 80%)
  - Track B / Path 2: Groq Vision LLM fallback (OCR confidence < 80%)
  - Unified Docling-schema AST output
"""
from smart_triage.orchestrator import (
    SmartTriageOrchestrator,
    ExecutionPath,
    PDFPLUMBER_AVAILABLE,
    PYMUPDF_AVAILABLE,
    PDFIUM_AVAILABLE,
    PADDLE_AVAILABLE,
)
from smart_triage.ast_compiler import ASTCompiler

__all__ = [
    "SmartTriageOrchestrator",
    "ExecutionPath",
    "ASTCompiler",
    "PDFPLUMBER_AVAILABLE",
    "PYMUPDF_AVAILABLE",
    "PDFIUM_AVAILABLE",
    "PADDLE_AVAILABLE",
]
