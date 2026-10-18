"""Command-line ingestion: bulk-load every document in a folder.

Useful for pre-loading the knowledge base before a demo instead of uploading
files one by one in the UI.

Usage:
    python ingest.py data/            # add the folder's documents to the index
    python ingest.py --reset data/    # wipe the index first, then ingest (clean rebuild)

Use --reset whenever you have removed or replaced files in data/. A plain
ingest only ADDS documents; it never removes ones that are already indexed, so
without --reset you can end up with stale chunks from deleted files.

README files are skipped automatically (they are documentation, not knowledge).
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.loaders import SUPPORTED_EXTENSIONS, extract_text
from app.vector_store import VectorStore

# Filenames to never ingest (documentation, not knowledge).
SKIP_NAMES = {"readme.md", "readme.txt"}


def main(folder: str, reset: bool) -> None:
    path = Path(folder)
    if not path.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    store = VectorStore()

    if reset:
        print("Resetting knowledge base (deleting all existing chunks)...")
        store.reset()

    files = [
        p
        for p in path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.name.lower() not in SKIP_NAMES
    ]

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
    print(f"Indexed documents: {store.list_sources()}")


if __name__ == "__main__":
    args = sys.argv[1:]
    reset = False
    if "--reset" in args:
        reset = True
        args.remove("--reset")
    folder = args[0] if args else "data"
    main(folder, reset)
