"""Command-line ingestion: bulk-load every document in a folder.

Useful for pre-loading the knowledge base before a demo instead of uploading
files one by one in the UI.

Usage:
    python ingest.py data/
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.loaders import SUPPORTED_EXTENSIONS, extract_text
from app.vector_store import VectorStore


def main(folder: str) -> None:
    path = Path(folder)
    if not path.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    store = VectorStore()
    files = [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"No supported documents found in {folder}.")
        print(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    total_chunks = 0
    for f in files:
        text = extract_text(f.name, f.read_bytes())
        if not text.strip():
            print(f"  skipped (no text): {f.name}")
            continue
        n = store.add_document(f.name, text)
        total_chunks += n
        print(f"  added {f.name}  ({n} chunks)")

    print(f"\nDone. {len(files)} file(s) processed, {total_chunks} chunks indexed.")
    print(f"Knowledge base now holds {store.count()} chunks total.")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(folder)
