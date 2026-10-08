import pytest
from unittest.mock import patch, MagicMock
from smart_triage.orchestrator import SmartTriageOrchestrator

@pytest.fixture
def orchestrator():
    return SmartTriageOrchestrator()

@patch("smart_triage.orchestrator.cv2.imread")
@patch.object(SmartTriageOrchestrator, "classify_and_route_image")
def test_7_stage_pipeline_text_density_override(mock_classify, mock_imread, orchestrator, text_heavy_image):
    # Mock imread to return our text heavy image
    mock_imread.return_value = text_heavy_image
    
    # Ensure local text density classifier is loaded
    if orchestrator._text_density_clf is None:
        pytest.skip("OCR Engine not available for testing")
        
    # Mock multi-strategy engine to return a highly confident result
    mock_ocr_result = MagicMock()
    mock_ocr_result.confidence = 90.0
    mock_ocr_result.word_count = 20
    mock_ocr_result.text = "This is a very confident text document extraction."
    mock_ocr_result.engine = "mock_tesseract"
    mock_ocr_result.strategy.name = "mock_strategy"
    
    orchestrator._multi_ocr_engine.run = MagicMock(return_value=mock_ocr_result)
    
    # The VLM should not be called at all because local density + local OCR is confident
    result = orchestrator._run_track_b(text_heavy_image, "dummy.jpg", "job123")
    
    mock_classify.assert_not_called()
    assert result["execution_path"] == "PATH_1"
    assert result["pages"][0]["_ocr_confidence"] == 90.0
    assert result["pages"][0]["extracted_text"] == "This is a very confident text document extraction."

@patch("smart_triage.orchestrator.cv2.imread")
@patch.object(SmartTriageOrchestrator, "classify_and_route_image")
def test_7_stage_pipeline_vlm_override(mock_classify, mock_imread, orchestrator, dummy_image):
    # Mock imread to return dummy image (ambiguous text density)
    mock_imread.return_value = dummy_image
    
    if orchestrator._text_density_clf is None:
        pytest.skip("OCR Engine not available for testing")
        
    # Make text density classifier return ambiguous
    orchestrator._text_density_clf.score = MagicMock(return_value=(0.10, {}))
    orchestrator._text_density_clf.classify = MagicMock(return_value="ambiguous")
    
    # Make VLM say general_image
    mock_classify.return_value = {"type": "general_image", "detected_text": ""}
    
    # BUT make local OCR find confident text
    mock_ocr_result = MagicMock()
    mock_ocr_result.confidence = 50.0  # Between 45.0 (override) and 55.0 (skip VLM)
    mock_ocr_result.word_count = 15
    mock_ocr_result.text = "Actually this is a text document."
    mock_ocr_result.engine = "mock_tesseract"
    mock_ocr_result.strategy.name = "mock_strategy"
    
    orchestrator._multi_ocr_engine.run = MagicMock(return_value=mock_ocr_result)
    
    result = orchestrator._run_track_b(dummy_image, "dummy.jpg", "job123")
    
    # VLM was called
    mock_classify.assert_called_once()
    
    # But result was overridden to text_document (since 50.0 is not > 50.0, path is PATH_2)
    assert result["execution_path"] == "PATH_2"
    assert result["pages"][0]["extracted_text"] == "Actually this is a text document."
    assert result["pages"][0]["_ocr_confidence"] == 50.0
