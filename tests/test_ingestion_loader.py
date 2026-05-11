from pathlib import Path

from src.ingestion.loader import load_pdfs


def test_empty_data_dir_returns_empty(tmp_path: Path) -> None:
    docs = load_pdfs(str(tmp_path))
    assert docs == []

