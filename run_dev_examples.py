"""Run the development example scenarios (D01–D08) through the assistant.

Usage:
    python run_dev_examples.py examples/AFS_Examples_and_Reference.docx

Writes answers to outputs/dev/ so you can sanity-check answer quality against
the sample answers in the document before the real evaluation. Uses the same
pipeline as the live chat (no hard-coding).
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.assistant import SpectroscopyAssistant
from app.scenarios import parse_scenarios, run_scenarios
from app.vector_store import VectorStore

OUTPUT_DIR = Path("outputs/dev")


def main(path_str: str) -> None:
    path = Path(path_str)
    if not path.is_file():
        print(f"File not found: {path_str}")
        sys.exit(1)

    store = VectorStore()
    if store.count() == 0:
        print(
            "The knowledge base is empty. Ingest the AFS materials first:\n"
            "    python ingest.py data/"
        )
        sys.exit(1)

    scenarios = parse_scenarios(path.name, path.read_bytes())
    if not scenarios:
        print("No scenarios found in the file.")
        sys.exit(1)

    print(f"Found {len(scenarios)} scenario(s): "
          f"{', '.join(s.scenario_id for s in scenarios)}")

    assistant = SpectroscopyAssistant(store)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for scenario, markdown in run_scenarios(assistant, scenarios):
        out = OUTPUT_DIR / f"{scenario.scenario_id}.md"
        out.write_text(markdown, encoding="utf-8")
        print(f"  wrote {out}")

    print(f"\nDone. Answers written to {OUTPUT_DIR}/.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_dev_examples.py <examples_file>")
        sys.exit(1)
    main(sys.argv[1])
