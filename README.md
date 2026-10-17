# 🔬 Spectroscopy Use-Case Assessment Assistant

An AI assistant that helps an **AFS spectroscopy expert** make a *first
technical assessment* of a new industrial customer problem.

Industrial customers usually know **what** they need to monitor (concentration,
reaction progress, contamination, moisture, product quality…) but not **which**
spectroscopy method, spectral range, fiber probe, or measurement setup to test
first. This assistant reads a short use case and helps the expert:

- restate and clarify the use case,
- identify **missing information** to ask the customer,
- recommend one or more promising **spectroscopy approaches**,
- **explain** each recommendation transparently, and
- propose a **simple first feasibility experiment**.

> It does **not** replace the expert. It structures the first customer
> discussion and makes recommendations that are easy to review.

## Why this design

The assistant is a **Retrieval-Augmented Generation (RAG)** chatbot. This
directly satisfies the challenge requirements:

| Requirement | How it's met |
| --- | --- |
| **Scalable** — add new documents over time | Upload PDFs/Word/Excel in the UI (or bulk-load a folder). Each doc is chunked, embedded, and indexed. More documents → smarter assistant, no code changes. |
| **No hallucination** | The model may answer **only** from retrieved document chunks. If nothing relevant is found, it refuses *before* even calling the LLM. Temperature is low (0.2). |
| **Cites its sources** | Every answer requires inline `[source: filename]` citations, and the UI shows the exact chunks each answer was built from. |
| **Asks when unsure** | If the context is insufficient, it asks follow-up questions or says it cannot recommend yet — rather than guessing. |

## Tech stack (all free)

- **Streamlit** — chat UI + document upload
- **Google Gemini** (`gemini-2.5-flash`, free API tier) — the language/reasoning layer
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local, free embeddings (no API cost)
- **ChromaDB** — local persistent vector store

## Architecture

```
          ┌──────────────── Streamlit chat app ────────────────┐
          │  chat window              upload (PDF/Word/Excel)   │
          └──────────┬───────────────────────────┬─────────────┘
                     │ ask                        │ add document
                     ▼                            ▼
         ┌───────────────────────┐    ┌──────────────────────────┐
         │  assistant.py          │    │  loaders.py + vector_store│
         │  retrieve top-k chunks │    │  extract text → chunk →   │
         │  → strict prompt →     │    │  embed → store            │
         │  Gemini → cited answer │    └────────────┬──────────────┘
         └───────────┬────────────┘                 │
                     │        retrieve               ▼
                     └──────────────────────► ChromaDB (local)
```

Source files:

- `app/loaders.py` — extract text from PDF / Word / Excel / text
- `app/vector_store.py` — chunking, embedding, ChromaDB storage & retrieval
- `app/assistant.py` — RAG pipeline + anti-hallucination system prompt
- `app/config.py` — all tunable settings
- `streamlit_app.py` — the chat UI
- `ingest.py` — bulk-load a folder of documents from the command line

## Setup & run

### 1. Get a free Gemini API key
Create one at <https://aistudio.google.com/apikey> (free tier, no billing needed).

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key
```bash
cp .env.example .env
# then edit .env and paste your key into GEMINI_API_KEY
```

### 4. (Optional) Pre-load documents
Drop files into `data/` and run:
```bash
python ingest.py data/
```
Or just upload them in the app sidebar once it's running.

### 5. Launch the app
```bash
streamlit run streamlit_app.py
```
Open the URL it prints (usually <http://localhost:8501>).

## How to demo

1. Upload a couple of spectroscopy reference documents in the sidebar.
2. Paste one of your customer use cases into the chat, e.g.
   *"A customer wants to monitor moisture content in a powder blend during
   production."*
3. Watch the assistant restate the case, list missing info, recommend
   method(s) with citations, and suggest a feasibility experiment.
4. Try a question with **no** supporting document to show it refuses / asks for
   more instead of hallucinating.

## Configuration knobs

All in `app/config.py`:

- `GEMINI_MODEL` — which Gemini model to use
- `TOP_K` — how many chunks to retrieve per question
- `MAX_DISTANCE` — relevance cutoff; higher = more permissive retrieval
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — document chunking granularity

## Limitations (be transparent with judges)

- Quality depends entirely on the uploaded documents — it's only as good as the
  knowledge base.
- Local embeddings are small/fast but not state-of-the-art; swap in a larger
  embedding model if needed.
- First run downloads the embedding model (~90 MB) and needs internet for the
  Gemini API.
- This is a prototype for a first assessment, not a validated engineering tool.

## Possible next iterations

- Highlight the exact quoted sentence used for each claim.
- Add a structured "intake form" that pre-fills the missing-information checklist.
- Support images/spectra and figure captions.
- Add evaluation on a set of known use cases to measure answer quality.
