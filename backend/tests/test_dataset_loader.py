import os
import json
import pytest
from app.ingestion.dataset_loader import DatasetLoader
from app.config import settings

def test_dataset_loader_initialization():
    loader = DatasetLoader(dataset_name="dummy/dataset")
    assert loader.dataset_name == "dummy/dataset"
    assert loader.dataset is None

def test_dataset_schema():
    """
    Validates dataset schema if report exists
    """
    report_path = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        assert "splits" in report
        assert len(report["splits"]) > 0
        assert "columns" in report
        
        # Basic check on a split that has columns
        has_columns = False
        for split in report["splits"]:
            if split in report["columns"]:
                assert len(report["columns"][split]) > 0
                has_columns = True
                break
        assert has_columns, "No split had columns defined"
        assert report["records_per_split"][split] > 0
    else:
        pytest.skip("Dataset report not generated yet. Run scripts/inspect_dataset.py first.")
