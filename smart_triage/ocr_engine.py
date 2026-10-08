"""
smart_triage/ocr_engine.py
==========================
Production-quality Multi-Strategy OCR Engine.

Architecture (SOLID, Clean Architecture):
  - OCRResult        : immutable dataclass for a single OCR attempt result
  - PreprocessingStrategy : enum of available preprocessing techniques
  - ImagePreprocessor : stateless service that applies one strategy to an image
  - MultiStrategyOCREngine : orchestrates strategy cascade with early-exit
  - LocalTextDensityClassifier : OpenCV-based pre-VLM text-density screen
  - DocumentTypeClassifier : keyword + heuristic doc-type inference from text

Design Decisions:
  - All methods are pure functions (no side effects on instance state).
  - Each strategy is tried in isolation; the best result is returned.
  - pytesseract is optional; system falls back to EasyOCR when unavailable.
  - All timings are logged at INFO level for diagnostics.
  - Dependency injection: pass the OCR backend as a callable, or use default.
"""

from __future__ import annotations

import logging
import time
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional backends — graceful degradation
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from pytesseract import Output as TessOutput
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed — OCR will use EasyOCR fallback only.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed — OCR fallback not available.")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PreprocessingStrategy(Enum):
    """Ordered list of preprocessing strategies, from cheapest to most aggressive."""
    GRAYSCALE_ADAPTIVE    = auto()  # Grayscale + Adaptive Gaussian threshold (fast baseline)
    GRAYSCALE_OTSU        = auto()  # Grayscale + Otsu threshold
    CLAHE_ADAPTIVE        = auto()  # CLAHE contrast enhancement + Adaptive threshold
    CLAHE_OTSU            = auto()  # CLAHE + Otsu (good for low-contrast images)
    MORPHOLOGICAL         = auto()  # Dilation/erosion to connect broken characters
    DPI_ENHANCE_OTSU      = auto()  # 2x upscale + Otsu (best for small text)
    BILATERAL_ADAPTIVE    = auto()  # Bilateral filter (noise preserving) + Adaptive
    INVERT_ADAPTIVE       = auto()  # Inverted image (white-on-dark text)
    DESKEW_ADAPTIVE       = auto()  # Deskew correction + Adaptive threshold


@dataclass(frozen=True)
class OCRResult:
    """
    Immutable container for a single OCR attempt result.

    Attributes:
        text: The extracted text string.
        confidence: Average word confidence (0.0–100.0).
        word_count: Number of distinct words detected.
        strategy: The preprocessing strategy that produced this result.
        processing_time_ms: Wall-clock time of the full strategy execution.
        engine: Name of the OCR engine ('tesseract' or 'easyocr').
        raw_data: Optional raw engine output for detailed diagnostics.
    """
    text: str
    confidence: float
    word_count: int
    strategy: PreprocessingStrategy
    processing_time_ms: float
    engine: str
    raw_data: dict = field(default_factory=dict, compare=False)

    @property
    def is_meaningful(self) -> bool:
        """True when the result contains enough text to be useful."""
        return self.word_count >= 3 and len(self.text.strip()) >= 10

    @property
    def quality_score(self) -> float:
        """Composite quality score blending confidence and text length."""
        length_bonus = min(len(self.text.strip()) / 500.0, 1.0) * 10.0
        return self.confidence + length_bonus


@dataclass
class ClassificationResult:
    """Result of the full image classification pipeline."""
    classification: str           # 'text_document' | 'general_image'
    text_density_score: float     # 0.0 – 1.0 (from LocalTextDensityClassifier)
    ocr_result: Optional[OCRResult]
    document_type: Optional[str]  # 'invoice', 'receipt', etc.
    reasoning: str                # Human-readable explanation
    processing_time_ms: float


# ---------------------------------------------------------------------------
# ImagePreprocessor — pure stateless service
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Stateless service that applies a named preprocessing strategy to an image.

    All methods accept BGR numpy arrays (OpenCV native format) and return
    a processed grayscale image ready for OCR.
    """

    @staticmethod
    def apply(image_bgr: np.ndarray, strategy: PreprocessingStrategy) -> np.ndarray:
        """
        Dispatch to the correct preprocessing method.

        Args:
            image_bgr: Input image in BGR format (OpenCV default).
            strategy: Which preprocessing strategy to apply.

        Returns:
            Preprocessed grayscale image.
        """
        dispatch = {
            PreprocessingStrategy.GRAYSCALE_ADAPTIVE : ImagePreprocessor._grayscale_adaptive,
            PreprocessingStrategy.GRAYSCALE_OTSU     : ImagePreprocessor._grayscale_otsu,
            PreprocessingStrategy.CLAHE_ADAPTIVE     : ImagePreprocessor._clahe_adaptive,
            PreprocessingStrategy.CLAHE_OTSU         : ImagePreprocessor._clahe_otsu,
            PreprocessingStrategy.MORPHOLOGICAL      : ImagePreprocessor._morphological,
            PreprocessingStrategy.DPI_ENHANCE_OTSU   : ImagePreprocessor._dpi_enhance_otsu,
            PreprocessingStrategy.BILATERAL_ADAPTIVE : ImagePreprocessor._bilateral_adaptive,
            PreprocessingStrategy.INVERT_ADAPTIVE    : ImagePreprocessor._invert_adaptive,
            PreprocessingStrategy.DESKEW_ADAPTIVE    : ImagePreprocessor._deskew_adaptive,
        }
        fn = dispatch.get(strategy)
        if fn is None:
            raise ValueError(f"Unknown strategy: {strategy}")
        return fn(image_bgr)

    # ── Internal strategy implementations ──────────────────────────────────

    @staticmethod
    def _to_gray_denoised(image_bgr: np.ndarray) -> np.ndarray:
        """Shared first step: grayscale + Gaussian noise reduction."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def _grayscale_adaptive(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 1: Grayscale + Gaussian blur + Adaptive threshold (fast baseline)."""
        gray = ImagePreprocessor._to_gray_denoised(image_bgr)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=2
        )

    @staticmethod
    def _grayscale_otsu(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 2: Grayscale + Gaussian blur + Otsu global threshold."""
        gray = ImagePreprocessor._to_gray_denoised(image_bgr)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _clahe_adaptive(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 3: CLAHE contrast enhancement + Adaptive threshold."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=2
        )

    @staticmethod
    def _clahe_otsu(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 4: CLAHE + Otsu — good for low-contrast images."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _morphological(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 5: Morphological opening to connect broken characters."""
        gray = ImagePreprocessor._to_gray_denoised(image_bgr)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cv2.bitwise_not(opened)

    @staticmethod
    def _dpi_enhance_otsu(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 6: 2× upscaling + Otsu — best for small or low-DPI text."""
        h, w = image_bgr.shape[:2]
        upscaled = cv2.resize(
            image_bgr, (w * 2, h * 2),
            interpolation=cv2.INTER_CUBIC
        )
        gray = ImagePreprocessor._to_gray_denoised(upscaled)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _bilateral_adaptive(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 7: Bilateral filter (edge-preserving denoising) + Adaptive threshold."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        return cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=2
        )

    @staticmethod
    def _invert_adaptive(image_bgr: np.ndarray) -> np.ndarray:
        """Strategy 8: Inverted image — for white-text-on-dark-background docs."""
        gray = ImagePreprocessor._to_gray_denoised(image_bgr)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=2
        )
        return cv2.bitwise_not(thresh)

    @staticmethod
    def _deskew_adaptive(image_bgr: np.ndarray) -> np.ndarray:
        """
        Strategy 9: Deskew correction + Adaptive threshold.
        Uses Hough line transform to detect and correct document tilt.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        angle_deg = 0.0
        if lines is not None:
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = (theta - np.pi / 2) * 180 / np.pi
                if abs(angle) < 45:  # Ignore near-vertical lines
                    angles.append(angle)
            if angles:
                angle_deg = float(np.median(angles))

        if abs(angle_deg) > 0.5:
            h, w = image_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
            deskewed = cv2.warpAffine(
                image_bgr, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            logger.debug("Deskewed image by %.2f°", angle_deg)
        else:
            deskewed = image_bgr

        gray2 = ImagePreprocessor._to_gray_denoised(deskewed)
        return cv2.adaptiveThreshold(
            gray2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=2
        )


# ---------------------------------------------------------------------------
# MultiStrategyOCREngine
# ---------------------------------------------------------------------------

class MultiStrategyOCREngine:
    """
    Orchestrates multiple preprocessing strategies with confidence-gated early exit.

    The engine tries strategies in order. When a strategy achieves the
    configured confidence threshold, it returns immediately (early exit).
    If no strategy meets the threshold, it returns the best result found.

    Args:
        confidence_threshold: Minimum acceptable OCR confidence (0–100).
            Early exit is triggered when this is met.
        min_meaningful_words: Minimum word count to consider a result meaningful.
        easyocr_reader: Optional pre-initialized EasyOCR reader (avoids re-init).
    """

    # Default strategy cascade — ordered from fastest/cheapest to slowest/most aggressive
    DEFAULT_STRATEGY_CASCADE = [
        PreprocessingStrategy.GRAYSCALE_ADAPTIVE,
        PreprocessingStrategy.CLAHE_ADAPTIVE,
        PreprocessingStrategy.GRAYSCALE_OTSU,
        PreprocessingStrategy.BILATERAL_ADAPTIVE,
        PreprocessingStrategy.CLAHE_OTSU,
        PreprocessingStrategy.MORPHOLOGICAL,
        PreprocessingStrategy.DPI_ENHANCE_OTSU,
        PreprocessingStrategy.DESKEW_ADAPTIVE,
        PreprocessingStrategy.INVERT_ADAPTIVE,
    ]

    def __init__(
        self,
        confidence_threshold: float = 60.0,
        min_meaningful_words: int = 3,
        easyocr_reader=None,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_meaningful_words = min_meaningful_words
        self._easyocr_reader = easyocr_reader
        self._preprocessor = ImagePreprocessor()

    def run(
        self,
        image_bgr: np.ndarray,
        strategies: Optional[list[PreprocessingStrategy]] = None,
    ) -> OCRResult:
        """
        Run OCR with the strategy cascade.

        Args:
            image_bgr: Input image in BGR format.
            strategies: Custom strategy list, or None to use default cascade.

        Returns:
            The best OCRResult found across all strategies attempted.
        """
        cascade = strategies or self.DEFAULT_STRATEGY_CASCADE
        best: Optional[OCRResult] = None
        total_start = time.perf_counter()

        logger.info("OCR cascade starting — %d strategies, threshold=%.1f%%",
                    len(cascade), self.confidence_threshold)

        for strategy in cascade:
            t_start = time.perf_counter()

            try:
                processed = ImagePreprocessor.apply(image_bgr, strategy)
                result = self._run_ocr_on_processed(processed, strategy, t_start)
            except Exception as exc:
                elapsed = (time.perf_counter() - t_start) * 1000
                logger.warning("Strategy %s failed: %s (%.1fms)", strategy.name, exc, elapsed)
                continue

            logger.info(
                "  Strategy %-25s → conf=%.1f%% | words=%d | time=%.1fms",
                strategy.name, result.confidence, result.word_count, result.processing_time_ms
            )

            # Track best result
            if best is None or result.quality_score > best.quality_score:
                best = result

            # Early exit if threshold met and result is meaningful
            if (result.confidence >= self.confidence_threshold
                    and result.word_count >= self.min_meaningful_words):
                logger.info("  Early exit: confidence %.1f%% >= threshold %.1f%%",
                            result.confidence, self.confidence_threshold)
                break

        total_ms = (time.perf_counter() - total_start) * 1000
        if best is None:
            best = OCRResult(
                text="", confidence=0.0, word_count=0,
                strategy=cascade[0], processing_time_ms=total_ms, engine="none"
            )

        logger.info(
            "OCR cascade complete — best: conf=%.1f%% words=%d strategy=%s total=%.1fms",
            best.confidence, best.word_count, best.strategy.name, total_ms
        )
        return best

    def _run_ocr_on_processed(
        self,
        processed: np.ndarray,
        strategy: PreprocessingStrategy,
        t_start: float,
    ) -> OCRResult:
        """Run OCR on a preprocessed grayscale image using available backend."""
        if TESSERACT_AVAILABLE:
            return self._run_tesseract(processed, strategy, t_start)
        elif EASYOCR_AVAILABLE:
            return self._run_easyocr(processed, strategy, t_start)
        else:
            elapsed = (time.perf_counter() - t_start) * 1000
            return OCRResult(
                text="", confidence=0.0, word_count=0,
                strategy=strategy, processing_time_ms=elapsed, engine="none",
                raw_data={"error": "No OCR backend available"}
            )

    def _run_tesseract(
        self,
        processed: np.ndarray,
        strategy: PreprocessingStrategy,
        t_start: float,
    ) -> OCRResult:
        """Run pytesseract on preprocessed image; compute per-word confidence."""
        config = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            processed,
            config=config,
            output_type=TessOutput.DICT,
        )

        texts, confs = [], []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = data["conf"][i]
            if word and conf > 0:  # conf=-1 means no data
                texts.append(word)
                confs.append(float(conf))

        extracted = " ".join(texts)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        elapsed = (time.perf_counter() - t_start) * 1000

        return OCRResult(
            text=extracted,
            confidence=avg_conf,
            word_count=len(texts),
            strategy=strategy,
            processing_time_ms=elapsed,
            engine="tesseract",
            raw_data={"word_confs": confs[:20]},  # limit raw dump size
        )

    def _run_easyocr(
        self,
        processed: np.ndarray,
        strategy: PreprocessingStrategy,
        t_start: float,
    ) -> OCRResult:
        """Run EasyOCR on preprocessed image as fallback."""
        if self._easyocr_reader is None:
            import easyocr as _easyocr
            self._easyocr_reader = _easyocr.Reader(["en"], gpu=False)

        results = self._easyocr_reader.readtext(processed, detail=1)
        texts, confs = [], []
        for (_bbox, text, conf) in results:
            if text.strip() and conf > 0.1:
                texts.append(text.strip())
                confs.append(conf * 100.0)  # normalize to 0–100

        extracted = " ".join(texts)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        elapsed = (time.perf_counter() - t_start) * 1000

        return OCRResult(
            text=extracted,
            confidence=avg_conf,
            word_count=len(texts),
            strategy=strategy,
            processing_time_ms=elapsed,
            engine="easyocr",
        )


# ---------------------------------------------------------------------------
# LocalTextDensityClassifier
# ---------------------------------------------------------------------------

class LocalTextDensityClassifier:
    """
    Lightweight OpenCV-based classifier to pre-screen images for text content.

    Runs in <20ms on CPU. Used as a gate BEFORE the expensive VLM router call:
      - Score > HIGH_THRESHOLD  → text document (skip VLM)
      - Score < LOW_THRESHOLD   → general image (skip VLM)
      - Score in between        → ambiguous → escalate to VLM

    Signal composition:
      1. Horizontal morphological text-line coverage  (weight 0.45)
      2. Canny edge density normalized by image area  (weight 0.25)
      3. Connected component (character proxy) density(weight 0.20)
      4. Aspect ratio of typical text regions         (weight 0.10)
    """

    # Default thresholds (tunable via environment variables)
    HIGH_THRESHOLD = 0.18   # Above this → confidently text document
    LOW_THRESHOLD  = 0.04   # Below this → confidently general image
    # Between 0.04 and 0.18 → ambiguous, escalate to VLM

    W_HLINES  = 0.45
    W_EDGES   = 0.25
    W_COMPS   = 0.20
    W_ASPECT  = 0.10

    def score(self, image_bgr: np.ndarray) -> tuple[float, dict]:
        """
        Compute text density score for the image.

        Args:
            image_bgr: Input image in BGR format.

        Returns:
            (score, signals) where score ∈ [0.0, 1.0] and signals is a
            breakdown dict for diagnostics.
        """
        t_start = time.perf_counter()
        h, w = image_bgr.shape[:2]
        page_area = float(h * w)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ── Signal 1: Horizontal text-line morphology ─────────────────────
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 15, 20), 2))
        hlines  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, hkernel)
        hline_contours, _ = cv2.findContours(hlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter: reasonable text-line height range (0.5%–5% of image height)
        min_h, max_h = h * 0.005, h * 0.05
        text_lines = [
            c for c in hline_contours
            if min_h <= cv2.boundingRect(c)[3] <= max_h
            and cv2.boundingRect(c)[2] >= w * 0.05  # at least 5% width
        ]
        hline_score = min(len(text_lines) / 20.0, 1.0)

        # ── Signal 2: Canny edge density ──────────────────────────────────
        edges = cv2.Canny(blurred, 50, 150)
        edge_score = min(cv2.countNonZero(edges) / (page_area * 0.08), 1.0)

        # ── Signal 3: Connected components as character proxies ───────────
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        # Filter: components with character-like size
        char_min_area = max(4, page_area * 0.00001)
        char_max_area = page_area * 0.01
        char_comps = [
            i for i in range(1, n_labels)
            if char_min_area <= stats[i, cv2.CC_STAT_AREA] <= char_max_area
        ]
        comp_score = min(len(char_comps) / 300.0, 1.0)

        # ── Signal 4: Aspect ratio of detected text-line regions ──────────
        # Text lines tend to be wider than tall (aspect ratio > 3:1)
        aspect_scores = []
        for c in text_lines:
            _, _, lw, lh = cv2.boundingRect(c)
            if lh > 0:
                ar = lw / lh
                aspect_scores.append(min(ar / 15.0, 1.0))
        aspect_score = float(np.mean(aspect_scores)) if aspect_scores else 0.0

        # ── Composite score ───────────────────────────────────────────────
        composite = (
            self.W_HLINES * hline_score
            + self.W_EDGES  * edge_score
            + self.W_COMPS  * comp_score
            + self.W_ASPECT * aspect_score
        )
        composite = round(min(composite, 1.0), 4)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        signals = {
            "hline_score"  : round(hline_score, 4),
            "edge_score"   : round(edge_score, 4),
            "comp_score"   : round(comp_score, 4),
            "aspect_score" : round(aspect_score, 4),
            "n_text_lines" : len(text_lines),
            "n_char_comps" : len(char_comps),
            "elapsed_ms"   : round(elapsed_ms, 2),
        }
        logger.info(
            "TextDensity score=%.4f | lines=%d comps=%d edges=%.3f aspect=%.3f (%.1fms)",
            composite, len(text_lines), len(char_comps), edge_score, aspect_score, elapsed_ms,
        )
        return composite, signals

    def classify(self, image_bgr: np.ndarray) -> str:
        """
        Classify image as 'text_document', 'general_image', or 'ambiguous'.

        Args:
            image_bgr: Input image in BGR format.

        Returns:
            One of: 'text_document', 'general_image', 'ambiguous'
        """
        score, _ = self.score(image_bgr)
        if score >= self.HIGH_THRESHOLD:
            return "text_document"
        elif score <= self.LOW_THRESHOLD:
            return "general_image"
        else:
            return "ambiguous"


# ---------------------------------------------------------------------------
# DocumentTypeClassifier
# ---------------------------------------------------------------------------

class DocumentTypeClassifier:
    """
    Classifies the type of document from extracted OCR text using keyword
    heuristics and pattern matching.

    Returns a document type label suitable for display and downstream routing.
    """

    # Pattern: (label, patterns_list) — ordered by specificity
    DOCUMENT_TYPES: list[tuple[str, list[str]]] = [
        ("Invoice",      ["invoice", "bill to", "ship to", "subtotal", "tax", "amount due", "payment due"]),
        ("Receipt",      ["receipt", "thank you for your purchase", "total paid", "cashier", "change due"]),
        ("ID Card",      ["date of birth", "d.o.b", "dob", "id no", "identification", "license no", "passport"]),
        ("Certificate",  ["certificate", "awarded to", "certify", "completion", "achievement"]),
        ("Contract",     ["agreement", "hereby agree", "terms and conditions", "party of", "whereas"]),
        ("Report",       ["executive summary", "introduction", "conclusion", "findings", "methodology"]),
        ("Form",         ["please fill", "signature", "applicant", "date of application", "form no"]),
        ("Prescription", ["rx", "prescribed by", "dosage", "tablet", "capsule", "mg", "pharmacy"]),
        ("Resume",       ["work experience", "education", "skills", "objective", "references"]),
        ("Screenshot",   ["http", "www.", "menu", "file edit view", "settings"]),
        ("Product Label",["nutritional information", "ingredients", "net weight", "serving size", "mrp"]),
        ("Presentation", ["slide", "agenda", "key takeaway", "thank you", "questions?"]),
    ]

    def classify(self, text: str) -> Optional[str]:
        """
        Infer document type from text content.

        Args:
            text: Extracted OCR text.

        Returns:
            Document type label, or None if type cannot be determined.
        """
        if not text.strip():
            return None

        lower = text.lower()
        scores: dict[str, int] = {}

        for doc_type, keywords in self.DOCUMENT_TYPES:
            hits = sum(1 for kw in keywords if kw in lower)
            if hits > 0:
                scores[doc_type] = hits

        if not scores:
            return None

        # Return the type with the most keyword hits
        best_type = max(scores, key=lambda k: scores[k])
        logger.info("DocumentType classified as '%s' (hits=%d)", best_type, scores[best_type])
        return best_type
