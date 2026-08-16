import pytest
from app.chunking.passage_extractor import PassageExtractor

def test_normalize_text():
    raw = "  This is \t a   test. \n\n\n\n It has spaces.  "
    normalized = PassageExtractor.normalize_text(raw)
    assert normalized == "This is a test. \n\n It has spaces."

def test_extract_passages():
    record = {
        "query_id": "123",
        "query_type": "DESC",
        "source_lang": "eng",
        "target_lang": "hin",
        "Eng_Query": "test",
        "query": "test_hin",
        "passages": {
            "Translated_passages": ["Passage 1", "Passage 2"],
            "is_selected": [1, 0]
        }
    }
    
    passages = PassageExtractor.extract_passages(record)
    assert len(passages) == 2
    
    assert passages[0].text == "Passage 1"
    assert passages[0].is_selected is True
    assert passages[0].passage_index == 0
    assert passages[0].query_id == "123"
    
    assert passages[1].text == "Passage 2"
    assert passages[1].is_selected is False
    assert passages[1].passage_index == 1

def test_extract_passages_empty():
    record = {"query_id": "1", "passages": {}}
    assert len(PassageExtractor.extract_passages(record)) == 0
