"""Config path resolution and artifact validation.

The package is code-only: artifact paths come from env vars or are empty. These tests
exercise the resolution order and the missing-artifact diagnostics without any real
artifacts on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from concept_extraction.config import (
    EMBED_MODEL,
    PRODUCTION,
    BlockConfig,
    validate_artifacts,
)


def _clear(monkeypatch):
    for name in (
        "CONCEPT_EXTRACTION_GLOSSARY_DIR",
        "CONCEPT_EXTRACTION_CLASSIFIER_MODEL",
        "CONCEPT_EXTRACTION_EMBED_MODEL",
        "PAPER_PROCESSING_ML_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_unconfigured(monkeypatch):
    """With nothing set, the block must not silently point at a private checkout."""
    _clear(monkeypatch)
    cfg = BlockConfig()
    assert cfg.glossary_dir == ""
    assert cfg.classifier_model_path == ""
    assert cfg.embed_model == EMBED_MODEL


def test_explicit_env_wins_over_service_root(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("PAPER_PROCESSING_ML_ROOT", "/some/root")
    monkeypatch.setenv("CONCEPT_EXTRACTION_GLOSSARY_DIR", str(tmp_path / "glossary"))
    monkeypatch.setenv("CONCEPT_EXTRACTION_CLASSIFIER_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("CONCEPT_EXTRACTION_EMBED_MODEL", "local-model-name")
    cfg = BlockConfig()
    assert cfg.glossary_dir == str(tmp_path / "glossary")
    assert cfg.classifier_model_path == str(tmp_path / "model")
    assert cfg.embed_model == "local-model-name"


def test_service_root_derives_paths(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PAPER_PROCESSING_ML_ROOT", "/root")
    cfg = BlockConfig()
    assert cfg.glossary_dir == str(Path("/root") / "data" / "glossary")
    assert cfg.classifier_model_path == str(
        Path("/root") / "model_weights" / "onnx_minilm_scientific_concept_classifier_model"
    )


def test_validate_reports_unset_paths(monkeypatch):
    _clear(monkeypatch)
    problems = validate_artifacts(BlockConfig())
    assert any("glossary_dir" in p for p in problems)
    assert any("classifier_model_path" in p for p in problems)


def test_validate_reports_missing_files(monkeypatch, tmp_path):
    _clear(monkeypatch)
    gdir = tmp_path / "glossary"
    gdir.mkdir()
    mdir = tmp_path / "model"
    mdir.mkdir()
    cfg = BlockConfig(glossary_dir=str(gdir), classifier_model_path=str(mdir))
    problems = validate_artifacts(cfg)
    assert any("glossary_faiss.index" in p for p in problems)
    assert any("model.onnx" in p for p in problems)


def test_validate_passes_with_complete_artifacts(monkeypatch, tmp_path):
    _clear(monkeypatch)
    gdir = tmp_path / "glossary"
    gdir.mkdir()
    (gdir / "glossary_faiss.index").write_bytes(b"idx")
    (gdir / "glossary_faiss.json").write_text("[]")
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "model.onnx").write_bytes(b"onnx")
    cfg = BlockConfig(glossary_dir=str(gdir), classifier_model_path=str(mdir))
    assert validate_artifacts(cfg) == []


def test_production_is_immutable_and_fingerprints():
    import dataclasses

    cfg = PRODUCTION
    assert dataclasses.is_dataclass(cfg)
    assert isinstance(cfg.fingerprint(), str)
    assert len(cfg.fingerprint()) == 16
    assert cfg.fingerprint() == cfg.fingerprint()