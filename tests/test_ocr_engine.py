import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from smart_triage.ocr_engine import (
    ImagePreprocessor,
    PreprocessingStrategy,
    LocalTextDensityClassifier,
    DocumentTypeClassifier,
    MultiStrategyOCREngine,
    OCRResult
)

def test_image_preprocessor_grayscale_adaptive(dummy_image):
    processed = ImagePreprocessor.apply(dummy_image, PreprocessingStrategy.GRAYSCALE_ADAPTIVE)
    assert processed is not None
    assert len(processed.shape) == 2  # Grayscale image

def test_local_text_density_classifier_general_image(dummy_image):
    classifier = LocalTextDensityClassifier()
    score, signals = classifier.score(dummy_image)
    assert score == 0.0
    classification = classifier.classify(dummy_image)
    assert classification == "general_image"

def test_local_text_density_classifier_text_document(text_heavy_image):
    classifier = LocalTextDensityClassifier()
    score, signals = classifier.score(text_heavy_image)
    assert score > LocalTextDensityClassifier.HIGH_THRESHOLD
    classification = classifier.classify(text_heavy_image)
    assert classification == "text_document"

def test_document_type_classifier():
    classifier = DocumentTypeClassifier()
    assert classifier.classify("Tax Invoice 101 Subtotal $50") == "Invoice"
    assert classifier.classify("Thank you for your purchase. Total paid: $20") == "Receipt"
    assert classifier.classify("This is just a random text with no keywords.") is None

@patch("smart_triage.ocr_engine.TESSERACT_AVAILABLE", False)
@patch("smart_triage.ocr_engine.EASYOCR_AVAILABLE", False)
def test_multi_strategy_engine_no_backend(dummy_image):
    engine = MultiStrategyOCREngine(confidence_threshold=80.0)
    result = engine.run(dummy_image, strategies=[PreprocessingStrategy.GRAYSCALE_ADAPTIVE])
    assert result.confidence == 0.0
    assert result.engine == "none"

def test_ocr_result_quality_score():
    r1 = OCRResult("hello", 50.0, 1, PreprocessingStrategy.GRAYSCALE_OTSU, 10, "none")
    r2 = OCRResult("hello "*100, 50.0, 100, PreprocessingStrategy.GRAYSCALE_OTSU, 10, "none")
    assert r2.quality_score > r1.quality_score
