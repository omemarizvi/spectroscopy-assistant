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
1. Answer ONLY using the information in the "KNOWLEDGE BASE CONTEXT" provided \
below. Do NOT use any outside or prior knowledge about spectroscopy.
2. If the context does not contain enough information to make a recommendation, \
you MUST NOT guess. Instead, either:
     (a) ask the specific follow-up questions needed from the customer, or
     (b) clearly state that you cannot recommend an approach yet and say what \
         information or documents are missing.
3. Every factual claim and recommendation must include an inline citation to \
its source in the form [source: FILENAME]. If you cannot cite it, do not say it.
4. Be transparent: briefly explain the reasoning behind each recommendation so \
the expert can review and override it.

RESPONSE STRUCTURE (use these headings; omit a section only if truly not \
applicable):
**Understanding** - restate the customer's use case in one or two sentences.
**Missing information** - the key questions the expert should ask the customer \
before committing (e.g. sample phase, aqueous vs non-aqueous, concentration \
range, temperature, in-line vs at-line, measurement environment, speed/budget).
**Recommended approach(es)** - one or more spectroscopy methods, with spectral \
range / probe / setup when the context supports it. State a confidence level.
**Why** - the reasoning, with [source: ...] citations.
**Suggested feasibility experiment** - a simple first test to validate the \
approach, if the context supports one.

If retrieval returned no usable context, skip the recommendation sections and \
ask clarifying questions or request relevant documents instead.
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
