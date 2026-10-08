import pytest
import pandas as pd
import tempfile
import os
from smart_triage.orchestrator import SmartTriageOrchestrator

@pytest.fixture
def orchestrator():
    return SmartTriageOrchestrator()

@pytest.fixture
def multi_sheet_excel():
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_excel(writer, sheet_name="Data1", index=False)
        pd.DataFrame({"C": [5], "D": [6]}).to_excel(writer, sheet_name="Data2", index=False)
        pd.DataFrame(columns=["E", "F"]).to_excel(writer, sheet_name="Empty", index=False)
        
    yield path
    os.remove(path)

def test_extract_with_pandas_metadata(orchestrator, multi_sheet_excel):
    pages = orchestrator._extract_with_pandas(multi_sheet_excel, is_csv=False)
    
    assert len(pages) == 3
    
    # Check Sheet 1
    p1 = pages[0]
    assert p1["sheet_name"] == "Data1"
    assert p1["row_count"] == 2
    assert p1["col_count"] == 2
    assert not p1["is_empty"]
    assert len(p1["tables"]) == 1
    
    # Check Sheet 2
    p2 = pages[1]
    assert p2["sheet_name"] == "Data2"
    assert p2["row_count"] == 1
    assert p2["col_count"] == 2
    assert not p2["is_empty"]
    
    # Check Sheet 3 (Empty)
    p3 = pages[2]
    assert p3["sheet_name"] == "Empty"
    assert p3["row_count"] == 0
    assert p3["col_count"] == 2
    assert p3["is_empty"]
    assert len(p3["tables"]) == 0
