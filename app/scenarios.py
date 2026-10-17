"""Parse customer scenarios from the challenge input files and run each one
through the assistant.

Two input formats are supported, both handled the same way:
  * customer_scenarios_for_final_evaluation.txt  -> scenarios H01..H06
  * AFS_Examples_and_Reference.docx              -> scenarios D01..D08

Scenarios are detected by an ID marker at the start of a line, e.g. "H01",
"D03", optionally followed by ":", ")", "." or "-". Everything up to the next
ID marker is that scenario's text.

The SAME assistant pipeline used by the live chat is used here — there is no
case-specific logic — which satisfies the challenge's "no hard-coding" rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.assistant import SpectroscopyAssistant
from app.loaders import extract_text

# Matches a scenario id at the start of a line: H01, D08, etc.
_ID_PATTERN = re.compile(r"^\s*([HD]\d{1,2})\b[\s:.)\-]*", re.MULTILINE)


@dataclass
class Scenario:
    scenario_id: str   # e.g. "H01"
    text: str          # the customer description


def parse_scenarios(filename: str, data: bytes) -> list[Scenario]:
    """Extract individual scenarios from a .txt or .docx input file."""
    text = extract_text(filename, data)
    if not text.strip():
        return []

    matches = list(_ID_PATTERN.finditer(text))

    # If no explicit IDs are found, treat the whole file as one scenario.
    if not matches:
        return [Scenario(scenario_id="S01", text=text.strip())]

    scenarios: list[Scenario] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            scenarios.append(Scenario(scenario_id=m.group(1).upper(), text=body))
    return scenarios


def render_answer_markdown(scenario: Scenario, answer: str, sources) -> str:
    """Format one scenario's result as a committable Markdown document."""
    lines = [
        f"# Scenario {scenario.scenario_id}",
        "",
        "## Customer input",
        "",
        f"> {scenario.text.strip()}",
        "",
        "## Assistant assessment",
        "",
        answer.strip(),
        "",
        "## AFS sources used",
        "",
    ]
    if sources:
        for i, c in enumerate(sources, start=1):
            snippet = c.text.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "…"
            lines.append(f"{i}. **{c.source}** — \"{snippet}\"")
    else:
        lines.append("_No AFS sources were retrieved for this scenario._")
    lines.append("")
    return "\n".join(lines)


def run_scenarios(
    assistant: SpectroscopyAssistant, scenarios: list[Scenario]
) -> list[tuple[Scenario, str]]:
    """Run every scenario through the assistant. Returns (scenario, markdown)."""
    results = []
    for sc in scenarios:
        reply = assistant.ask(sc.text)
        md = render_answer_markdown(sc, reply.answer, reply.sources)
        results.append((sc, md))
    return results
