"""The RAG assistant: retrieve relevant chunks, then ask Gemini to answer
strictly from them.

The design goals (from the challenge) are enforced here:
  * No hallucination  -> the prompt forbids using outside knowledge, and if
    retrieval finds nothing relevant we don't even call the model.
  * Cite sources      -> the prompt requires inline [source] citations and we
    also return the raw retrieved chunks for display.
  * Ask when unsure   -> the prompt instructs the model to ask follow-up
    questions or decline when the context is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from google import genai
from google.genai import types

from app import config
from app.vector_store import RetrievedChunk, VectorStore

SYSTEM_PROMPT = """\
You are an assistant that helps an AFS spectroscopy expert make a FIRST \
technical assessment of a new industrial customer problem. You support the \
expert's first customer discussion; you do NOT replace the expert.

ABSOLUTE RULES
1. The supplied AFS materials (the "KNOWLEDGE BASE CONTEXT" below) are your \
PRIMARY source. Base technical recommendations on them. Do NOT invent \
spectroscopy facts that the context does not support.
2. You MAY use general external knowledge only to support reasoning, but if you \
rely on an external fact you MUST quote the exact passage you are relying on and \
label it clearly as external. AFS materials remain the main source.
3. If the context does not contain enough information to make a recommendation, \
you MUST NOT guess. Instead, either:
     (a) ask the specific clarification questions needed, or
     (b) clearly state that you cannot recommend an approach yet and say what \
         information or AFS material is missing.
4. Every factual claim grounded in AFS materials must include an inline citation \
in the form [source: FILENAME]. If you cannot cite it to AFS material or quote \
an external passage, do not state it as fact.
5. Be transparent: explain the reasoning so an AFS expert can review and \
override it, and be explicit about uncertainty.

RESPONSE STRUCTURE — use these six headings in this order (omit a section only \
if truly not applicable):

**1. Requirements & constraints** — the main process requirements and \
constraints extracted from the customer description (what is to be monitored, \
sample phase/state, environment, in-line vs at-line, speed, etc.).

**2. Clarification questions** — the key questions to ask the customer when \
information is missing (e.g. sample phase, aqueous vs non-aqueous, concentration \
range, temperature, measurement environment, required accuracy, budget).

**3. Recommended method(s)** — one or more spectroscopy methods or combinations \
worth testing, with the relevant spectral range when the context supports it. \
State a confidence level (low / medium / high).

**4. Probe / measurement concept** — a high-level probe and measurement setup \
idea (e.g. probe type, contact vs non-contact, immersion/transmission/\
reflectance), grounded in AFS materials where possible.

**5. Explanation, limitations & uncertainty** — the reasoning behind the \
recommendation, with [source: ...] citations; the known limitations; and an \
honest statement of what is uncertain or assumed.

**6. First feasibility-study plan** — a short, practical plan for the first \
experiment(s) to test the approach (what to measure, how to compare methods, \
what a positive result looks like).

If retrieval returned no usable AFS context, skip sections 3-6 and instead ask \
clarification questions or request the relevant AFS material.
"""


@dataclass
class AssistantReply:
    """What the assistant returns to the UI."""

    answer: str
    sources: list[RetrievedChunk]
    used_context: bool  # False when we refused due to no relevant context


class SpectroscopyAssistant:
    def __init__(self, store: VectorStore) -> None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add "
                "your free key from https://aistudio.google.com/apikey"
            )
        self._store = store
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def ask(self, question: str, history: list[dict] | None = None) -> AssistantReply:
        """Answer a question using retrieval-augmented generation."""
        retrieved = self._store.search(question, top_k=config.TOP_K)

        # Keep only chunks that are actually relevant (distance threshold).
        relevant = [c for c in retrieved if c.distance <= config.MAX_DISTANCE]

        # Guardrail: if nothing relevant was found, refuse without calling the
        # model. This is the strongest anti-hallucination protection.
        if not relevant:
            msg = (
                "I couldn't find anything in the current knowledge base that "
                "relates to this. I can't make a recommendation without "
                "supporting information.\n\n"
                "You can either:\n"
                "- upload a relevant document (PDF / Word / Excel), or\n"
                "- give me more detail about the use case so I can search again."
            )
            return AssistantReply(answer=msg, sources=[], used_context=False)

        context = self._build_context(relevant)
        prompt = self._build_prompt(question, context, history)

        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                # Low temperature keeps answers grounded and repeatable.
                temperature=0.2,
            ),
        )

        return AssistantReply(
            answer=response.text or "",
            sources=relevant,
            used_context=True,
        )

    # --- Prompt assembly ---------------------------------------------------
    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(
                f"--- Context {i} [source: {c.source}] ---\n{c.text.strip()}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_prompt(question: str, context: str, history: list[dict] | None) -> str:
        convo = ""
        if history:
            # Give the model recent turns so follow-up questions make sense.
            lines = [
                f"{turn['role'].upper()}: {turn['content']}" for turn in history[-6:]
            ]
            convo = "CONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"

        return (
            f"{convo}"
            "KNOWLEDGE BASE CONTEXT (the only information you may use):\n"
            f"{context}\n\n"
            f"CUSTOMER USE CASE / QUESTION:\n{question}\n\n"
            "Produce your assessment following the required response structure. "
            "Remember: cite sources inline, and if the context is insufficient, "
            "ask follow-up questions instead of guessing."
        )
