"""CLI behaviour: input parsing, config building, output shape, self-check.

These tests do not require the ML artifacts — they exercise the CLI plumbing. The real
end-to-end run is covered by `test_block_parity.py` (skipped without artifacts).
"""

from __future__ import annotations

import json

import pytest

from concept_extraction.cli import _build_config, _read_sentences, build_parser, main


def _run(args: list[str]) -> int:
    return main(args)


def test_read_sentences_json_object(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"sentences": ["a. b. c.", "x y z."]}))
    assert _read_sentences(p) == ["a. b. c.", "x y z."]


def test_read_sentences_json_list(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps(["one", "two"]))
    assert _read_sentences(p) == ["one", "two"]


def test_read_sentences_text(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text("line one\nline two\n")
    assert _read_sentences(p) == ["line one", "line two"]


def test_read_sentences_json_missing_key(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"nope": []}))
    with pytest.raises(ValueError):
        _read_sentences(p)


def test_build_config_empty(isolated_env):
    cfg = _build_config(build_parser().parse_args(["run", "x", "-o", "y"]))
    assert cfg.glossary_dir == ""
    assert cfg.classifier_model_path == ""
    assert cfg.top_n == -1
    assert cfg.scientific_concept_threshold == 0.60


def test_build_config_overrides(isolated_env):
    args = build_parser().parse_args(
        [
            "run",
            "x",
            "-o",
            "y",
            "--glossary-dir",
            "/g",
            "--classifier-model",
            "/m",
            "--embed-model",
            "e",
            "--top-n",
            "100",
            "--threshold",
            "0.5",
        ]
    )
    cfg = _build_config(args)
    assert cfg.glossary_dir == "/g"
    assert cfg.classifier_model_path == "/m"
    assert cfg.embed_model == "e"
    assert cfg.top_n == 100
    assert cfg.scientific_concept_threshold == 0.5


def test_run_fails_cleanly_without_artifacts(isolated_env, tmp_path, capsys):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"sentences": ["a b. c d."]}))
    out = tmp_path / "out.json"
    rc = _run(["run", str(inp), "-o", str(out)])
    assert rc == 2
    assert not out.exists()
    assert "artifact configuration problems" in capsys.readouterr().err


def test_self_check_reports_missing_artifacts(isolated_env, capsys):
    rc = _run(["self-check"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "glossary_dir" in captured.out
    assert "not configured" in captured.out


def test_self_check_reports_vendored_hash(isolated_env, capsys):
    rc = _run(["self-check"])
    captured = capsys.readouterr()
    assert "vendored chunker hash:  OK" in captured.out


def test_fetch_subcommand_single_pdf(isolated_env, tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This sentence has more than eight words in it easily.")
    pdf_path = pdf_dir / "paperA.pdf"
    doc.save(str(pdf_path))
    doc.close()

    out_dir = tmp_path / "sentences"
    rc = _run(["fetch", str(pdf_path), "-o", str(out_dir)])

    assert rc == 0
    data = json.loads((out_dir / "paperA.json").read_text())
    assert data["paper_id"] == "paperA"
    assert data["sentences"] == ["This sentence has more than eight words in it easily."]


def test_fetch_subcommand_pdf_dir(isolated_env, tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for name in ("paperA", "paperB"):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This sentence has more than eight words in it easily.")
        doc.save(str(pdf_dir / f"{name}.pdf"))
        doc.close()

    out_dir = tmp_path / "sentences"
    rc = _run(["fetch", str(pdf_dir), "-o", str(out_dir)])

    assert rc == 0
    assert (out_dir / "paperA.json").exists()
    assert (out_dir / "paperB.json").exists()


# --- conftest-style fixture: clear artifact env for every test ---------------------


@pytest.fixture
def isolated_env(monkeypatch):
    for name in (
        "CONCEPT_EXTRACTION_GLOSSARY_DIR",
        "CONCEPT_EXTRACTION_CLASSIFIER_MODEL",
        "CONCEPT_EXTRACTION_EMBED_MODEL",
        "PAPER_PROCESSING_ML_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    yield