from src.ingestion import pipeline


def test_process_all_uses_dynamic_chunking_by_default(monkeypatch) -> None:
    calls = {"dynamic": 0, "static": 0}

    class Doc:
        source_file = "doc.pdf"
        pages = ["Some text."]

    def fake_load_pdfs(_data_dir):
        return [Doc()]

    def fake_clean_text(text):
        return text

    def fake_dynamic_chunk_text(*args, **kwargs):
        calls["dynamic"] += 1
        return []

    def fake_chunk_text(*args, **kwargs):
        calls["static"] += 1
        return []

    monkeypatch.setattr(pipeline, "load_pdfs", fake_load_pdfs)
    monkeypatch.setattr(pipeline, "clean_text", fake_clean_text)
    monkeypatch.setattr(pipeline, "dynamic_chunk_text", fake_dynamic_chunk_text)
    monkeypatch.setattr(pipeline, "chunk_text", fake_chunk_text)

    pipeline.process_all("data")

    assert calls["dynamic"] == 1
    assert calls["static"] == 0
