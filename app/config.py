"""Central configuration for the Spectroscopy Assistant.

All tunable knobs live here so they are easy to find and adjust during the
hackathon demo. Values can be overridden with environment variables.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Gemini (LLM) ---------------------------------------------------------
# Free API key from https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Embeddings -----------------------------------------------------------
# Runs locally (free, no API cost). First run downloads the model (~90 MB).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Vector store ---------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "spectroscopy_knowledge"

# --- Retrieval tuning -----------------------------------------------------
# How many document chunks to retrieve per question.
TOP_K = 5
# Chunks with a similarity distance ABOVE this are treated as irrelevant.
# (Chroma returns cosine *distance*: 0 = identical, 2 = opposite.)
# If every retrieved chunk is above this threshold, the assistant refuses
# to answer rather than hallucinate.
MAX_DISTANCE = 1.1

# --- Chunking -------------------------------------------------------------
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # overlap keeps context across chunk boundaries
