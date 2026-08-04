"""Command-line interface for the standalone extraction block.

Subcommands:

    concept-extract run    <input> -o <out>   run the block over sentences
    concept-extract self-check               validate artifact paths + vendored hash

Input is a JSON file with `{"sentences": [...]}` or a plain text file (one sentence per
line; blank lines are retained as-is). Output is JSON with the same shape as the parsed
route's response, plus the provenance fields the block records.

Artifacts are resolved from the environment (see `config.py`); CLI flags override env.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

from .candidates.ablation import BY_NAME
from .config import PRODUCTION, BlockConfig, validate_artifacts
from .run import run_block

CSV_FIELDS = [
    "id",
    "name",
    "level",
    "definition",
    "semantic_score",
    "faiss_score",
    "relevance_score",
    "source",
    "definition_source",
    "paper_context",
]


def _read_sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            if "sentences" not in data:
                raise ValueError("JSON input must have a 'sentences' list")
            return list(data["sentences"])
        if isinstance(data, list):
            return list(data)
        raise ValueError("JSON input must be an object with 'sentences' or a list")
    return text.splitlines()


def _build_config(args: argparse.Namespace) -> BlockConfig:
    # Build fresh from the current environment (not `PRODUCTION`, whose env-derived
    # defaults were baked at import time), then apply CLI overrides.
    overrides: dict = {}
    for flag, field in (
        ("glossary_dir", "glossary_dir"),
        ("classifier_model", "classifier_model_path"),
        ("embed_model", "embed_model"),
        ("top_n", "top_n"),
        ("threshold", "scientific_concept_threshold"),
    ):
        val = getattr(args, flag, None)
        if val is not None:
            overrides[field] = val
    return replace(BlockConfig(), **overrides)


def _result_to_dict(result) -> dict:
    return {
        key: getattr(result, key)
        for key in (
            "total_sentences",
            "execution_time_seconds",
            "stage_1_unique_nps",
            "stage_2_glossary_matches",
            "stage_3_unmatched_nps_evaluated",
            "stage_3_onnx_classified_positive",
            "total_concepts_returned",
            "concepts",
            "config_fingerprint",
            "rung",
            "vendored_sha",
        )
    }


def _cmd_run(args: argparse.Namespace) -> int:
    sentences = _read_sentences(args.input)
    if not sentences:
        print("error: no sentences in input", file=sys.stderr)
        return 2

    cfg = _build_config(args)
    problems = validate_artifacts(cfg)
    if problems:
        print("artifact configuration problems:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Fix these (see README) before running.", file=sys.stderr)
        return 2

    rung = BY_NAME[args.rung]
    result = run_block(sentences, cfg=cfg, rung=rung)

    out = _result_to_dict(result)
    if args.output.suffix.lower() == ".csv":
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for c in result.concepts:
                writer.writerow({k: c.get(k) for k in CSV_FIELDS})
    else:
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {len(result.concepts)} concepts to {args.output}")
    return 0


def _cmd_self_check(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    problems = validate_artifacts(cfg)

    print("resolved artifact paths:")
    print(f"  glossary_dir:          {cfg.glossary_dir or '(not configured)'}")
    print(f"  classifier_model_path: {cfg.classifier_model_path or '(not configured)'}")
    print(f"  embed_model:           {cfg.embed_model}")
    print(f"  top_n:                 {cfg.top_n}")

    from .candidates.chunker import verify_vendored

    try:
        sha = verify_vendored()
        print(f"vendored chunker hash:  OK ({sha[:12]}...)")
    except Exception as exc:  # noqa: BLE001 — report any verification failure
        print(f"vendored chunker hash:  FAIL — {exc}")
        problems.append(str(exc))

    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall artifacts present.")
    return 0


def _add_artifact_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--glossary-dir",
        help="path to the glossary dir (overrides CONCEPT_EXTRACTION_GLOSSARY_DIR)",
    )
    p.add_argument(
        "--classifier-model",
        help="path to the ONNX classifier dir (overrides CONCEPT_EXTRACTION_CLASSIFIER_MODEL)",
    )
    p.add_argument(
        "--embed-model",
        help="embedder name or local dir (overrides CONCEPT_EXTRACTION_EMBED_MODEL)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        help="ranking cap (default -1 = uncapped)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="classifier scientific-concept threshold (default 0.60)",
    )
    p.add_argument(
        "--rung",
        default="A0",
        choices=sorted(BY_NAME),
        help="candidate-generator ablation rung (default A0 = production)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concept-extract",
        description="Standalone scientific concept extraction block (steps 1-4).",
    )
    _add_artifact_flags(parser)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run_p = sub.add_parser("run", help="run the block over sentences")
    _add_artifact_flags(run_p)
    run_p.add_argument("input", type=Path, help="JSON ({'sentences': [...]}) or text file")
    run_p.add_argument("-o", "--output", type=Path, required=True, help="output .json or .csv")
    run_p.set_defaults(func=_cmd_run)

    check_p = sub.add_parser("self-check", help="validate artifacts and vendored hash")
    _add_artifact_flags(check_p)
    check_p.set_defaults(func=_cmd_self_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())