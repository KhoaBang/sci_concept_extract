"""Paper fulltext -> sentences, faithful to production's `read_pdf` + `split_sentences`.

`read_pdf_text` is a direct port of `concept_extraction/pdf_reader.read_pdf`: markdown
extraction via `pymupdf4llm`, falling back to plain per-page `fitz` text on any failure —
not a reimplementation with different defaults. `sentenize` calls the existing
`candidates.pipeline.split_sentences` unchanged, so no new sentence-splitting logic is
introduced by this harness.

No network is involved: it reads PDFs the user provides locally (a single file or a
directory of PDFs). It never downloads anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz
import pymupdf4llm

from ..candidates.pipeline import split_sentences


def read_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        try:
            text = pymupdf4llm.to_markdown(doc, use_ocr=False)
        except Exception:
            text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    return text


def sentenize(text: str) -> list[str]:
    return split_sentences(text)


def cache_sentences(paper_id: str, sentences: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper_id}.json"
    out_path.write_text(json.dumps({"paper_id": paper_id, "sentences": sentences}, indent=1))
    return out_path


def process_pdf(pdf_path: Path, out_dir: Path) -> None:
    """Extract sentences from a single PDF file and cache them as `{stem}.json`."""
    paper_id = pdf_path.stem
    out_path = out_dir / f"{paper_id}.json"
    if out_path.exists():
        print(f"  {paper_id}: already sentenized, skipping")
        return
    try:
        text = read_pdf_text(pdf_path)
    except Exception as exc:
        print(f"  {paper_id}: failed to read PDF ({exc}), skipping")
        return
    sentences = sentenize(text)
    cache_sentences(paper_id, sentences, out_dir)
    print(f"  {paper_id}: {len(sentences)} sentences")


def process_all(pdf_dir: Path, out_dir: Path) -> None:
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        process_pdf(pdf_path, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert your local PDF(s) to per-paper sentence JSON (no network)."
    )
    ap.add_argument(
        "pdf",
        type=Path,
        help="a single .pdf file, or a directory of .pdf files (the PDF pool)",
    )
    ap.add_argument("-o", "--out", dest="out_dir", type=Path, required=True, help="output dir")
    args = ap.parse_args()

    if args.pdf.is_dir():
        process_all(args.pdf, args.out_dir)
    else:
        process_pdf(args.pdf, args.out_dir)


if __name__ == "__main__":
    main()