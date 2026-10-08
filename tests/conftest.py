import json
import os
import sys
import uuid

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Image fixtures (existing)
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_image():
    """Returns a dummy 100x100 BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def text_heavy_image():
    """Returns a synthetic image that looks like text to the text density classifier."""
    img = np.zeros((200, 500, 3), dtype=np.uint8)
    for y in range(20, 180, 20):
        img[y:y+10, 20:480] = (255, 255, 255)
    return img

# ---------------------------------------------------------------------------
# Gateway v2 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tmp_dirs(tmp_path_factory):
    """Create isolated temp directories for gateway tests."""
    base = tmp_path_factory.mktemp("gateway")
    dirs = {
        "upload":   str(base / "temp_uploads"),
        "pending":  str(base / "pending_review"),
        "final":    str(base / "final_database"),
        "rejected": str(base / "rejected"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


@pytest.fixture(scope="session")
def client(tmp_dirs, monkeypatch_session):
    """FastAPI TestClient for the gateway app with patched directories."""
    # Patch env so gateway uses our temp dirs
    monkeypatch_session.setenv("UPLOAD_DIR",   tmp_dirs["upload"])
    monkeypatch_session.setenv("PENDING_DIR",  tmp_dirs["pending"])
    monkeypatch_session.setenv("FINAL_DIR",    tmp_dirs["final"])
    monkeypatch_session.setenv("REJECTED_DIR", tmp_dirs["rejected"])

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    # Import lazily after env is patched
    from fastapi.testclient import TestClient
    # Import a lightweight version — skip Celery by mocking
    import unittest.mock as mock
    with mock.patch("worker.process_document_task") as mock_task:
        mock_task.delay.return_value = mock.MagicMock(id="test-celery-id")
        from lego1_gateway.main import app
        yield TestClient(app)


@pytest.fixture(scope="session")
def monkeypatch_session(request):
    """Session-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def sample_pdf(tmp_path):
    """Minimal valid PDF file for upload tests."""
    pdf = tmp_path / "sample.pdf"
    # Minimal PDF header
    pdf.write_bytes(b"%PDF-1.4\n%EOF")
    return pdf


@pytest.fixture
def committed_doc(tmp_dirs):
    """Write a committed JSON doc containing a known keyword for search tests."""
    job_id = str(uuid.uuid4())
    data = {
        "_job_id":      job_id,
        "_pipeline":    "fast_track_cpu",
        "_approved_at": "2026-08-01T10:00:00Z",
        "document_type": "invoice",
        "extracted_text": "This is a GREENCARE_TEST_KEYWORD for testing search.",
        "key_value_pairs": {"Invoice No": "TEST-001"},
        "tables": [],
    }
    path = os.path.join(tmp_dirs["final"], f"{job_id}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return {"job_id": job_id, "path": path}

