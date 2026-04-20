import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.ingestion.edgar import (
    chunk_text,
    clean_text,
    validate_filing,
    get_filings,
    save_filing,
)


def test_chunk_text_basic():
    words = ["word"] * 600
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 500 for c in chunks)


def test_chunk_text_overlap():
    words = [str(i) for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    first_chunk_end = chunks[0].split()[-50:]
    second_chunk_start = chunks[1].split()[:50]
    assert first_chunk_end == second_chunk_start


def test_validate_filing_empty():
    valid, reason = validate_filing("")
    assert not valid
    assert "empty" in reason


def test_validate_filing_too_short():
    valid, reason = validate_filing("too short text")
    assert not valid
    assert "short" in reason


def test_validate_filing_valid():
    text = " ".join(["word"] * 300)
    valid, reason = validate_filing(text)
    assert valid
    assert reason == "ok"


def test_save_filing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.edgar.RAW_DATA_DIR", tmp_path)
    chunks = ["chunk one", "chunk two"]
    filepath = save_filing("AAPL", "0001234-24-000001", "2024-01-01", chunks)
    assert filepath.exists()
    with open(filepath) as f:
        data = json.load(f)
    assert data["ticker"] == "AAPL"
    assert data["chunk_count"] == 2
    assert data["chunks"] == chunks


@patch("src.ingestion.edgar.requests.get")
def test_get_filings_returns_correct_form(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-K", "8-K"],
                "accessionNumber": ["0001-24-001", "0001-24-002", "0001-24-003"],
                "filingDate": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        }
    }
    mock_get.return_value = mock_response
    results = get_filings("0000320193", form_type="8-K", limit=5)
    assert len(results) == 2
    assert all(r["form"] == "8-K" for r in results)


def test_clean_text_removes_base64():
    noisy = "Normal text here. " + "A" * 150 + "== more text"
    cleaned = clean_text(noisy)
    assert "A" * 150 not in cleaned
    assert "Normal text" in cleaned

def test_clean_text_removes_short_lines():
    text = "Good sentence with real content here.\n{color: red;}\nAnother good sentence."
    cleaned = clean_text(text)
    assert "Good sentence" in cleaned