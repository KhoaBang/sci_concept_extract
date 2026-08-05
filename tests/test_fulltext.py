"""PDF -> text -> sentences. Uses synthetic in-memory PDFs (built with fitz) so this
needs no network and no real paper — only the parsing logic under test."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from concept_extraction.fetch.fulltext import (
    cache_sentences,
    process_all,
    process_pdf,
    read_pdf_text,
    sentenize,
)


def _make_pdf(tmp_path: Path, text: str, name: str = "sample") -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    path = tmp_path / f"{name}.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_read_pdf_text_extracts_content(tmp_path):
    pdf_path = _make_pdf(tmp_path, "The quick brown fox jumps over the lazy dog.")
    text = read_pdf_text(pdf_path)
    assert "quick brown fox" in text


def test_read_pdf_text_falls_back_when_markdown_extraction_fails(tmp_path):
    pdf_path = _make_pdf(tmp_path, "Fallback path sentence for the parsing test.")
    with patch(
        "concept_extraction.fetch.fulltext.pymupdf4llm.to_markdown",
        side_effect=RuntimeError("boom"),
    ):
        text = read_pdf_text(pdf_path)
    assert "Fallback path sentence" in text


def test_sentenize_drops_sentences_under_eight_words():
    text = "Short one. This sentence has more than eight words in it easily."
    assert sentenize(text) == ["This sentence has more than eight words in it easily."]


def test_cache_sentences_writes_expected_json(tmp_path):
    out_path = cache_sentences("paper123", ["a sentence here"], tmp_path)
    assert out_path == tmp_path / "paper123.json"
    assert json.loads(out_path.read_text()) == {
        "paper_id": "paper123",
        "sentences": ["a sentence here"],
    }


def test_process_all_skips_unreadable_pdf_and_continues(tmp_path, capsys):
    pdf_dir = tmp_path / "pdfs"
    out_dir = tmp_path / "sentences"
    pdf_dir.mkdir()

    # One valid PDF ("good") and one corrupt, unopenable file that looks like a PDF ("bad").
    good_path = _make_pdf(pdf_dir, "This sentence has more than eight words in it easily.", name="good")
    good_path.rename(pdf_dir / "good.pdf")
    (pdf_dir / "bad.pdf").write_bytes(b"not a real pdf")

    process_all(pdf_dir, out_dir)

    assert (out_dir / "good.json").exists()
    assert not (out_dir / "bad.json").exists()
    captured = capsys.readouterr()
    assert "bad" in captured.out
    assert "good" in captured.out


def test_process_all_skips_already_sentenized(tmp_path, capsys):
    pdf_dir = tmp_path / "pdfs"
    out_dir = tmp_path / "sentences"
    pdf_dir.mkdir()
    _make_pdf(pdf_dir, "This sentence has more than eight words in it easily.", name="good")

    process_all(pdf_dir, out_dir)
    process_all(pdf_dir, out_dir)

    captured = capsys.readouterr()
    assert "already sentenized, skipping" in captured.out


def test_process_pdf_single_file(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = _make_pdf(pdf_dir, "This sentence has more than eight words in it easily.", name="paperA")
    out_dir = tmp_path / "sentences"

    process_pdf(pdf_path, out_dir)

    assert (out_dir / "paperA.json").exists()
    data = json.loads((out_dir / "paperA.json").read_text())
    assert data["paper_id"] == "paperA"
    assert data["sentences"] == ["This sentence has more than eight words in it easily."]