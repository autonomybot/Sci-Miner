"""
Single place for every external connection and setting used by the
graph_generator scripts:

  * file / folder paths
  * the local LLM backend (LM Studio or Ollama)
  * the Neo4j database connection

Every value can be overridden with an environment variable, so no secrets
need to be written into the code. Example (PowerShell):

    $env:LLM_BACKEND = "ollama"
    $env:LLM_MODEL   = "llama3.1"
    $env:NEO4J_PASSWORD = "my-password"
"""

import os

import requests

# Repo root = parent of this folder, so paths work from any working directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Folder that holds the Fortran sources (searched recursively).
FORTRAN_FOLDER = os.getenv("FORTRAN_FOLDER", os.path.join(ROOT_DIR, "data", "input", "fortran"))
# Step 1 output: LLM-extracted variables/constants/equations.
# Named *_generated so the existing data/intermediate/math.txt is never overwritten;
# point MATH_TXT_PATH at math.txt to rebuild the CSV from it without calling the LLM.
MATH_TXT_PATH = os.getenv("MATH_TXT_PATH", os.path.join(ROOT_DIR, "data", "intermediate", "math_generated.txt"))
# Step 2 output: the single graph CSV (source,target,relationship), same format as
# data/output/e3sm.csv. Used by Cytoscape and by sci-kg_contructor.py (Neo4j).
EDGES_CSV_PATH = os.getenv("EDGES_CSV_PATH", os.path.join(ROOT_DIR, "data", "output", "e3sm_generated.csv"))
# Name of the root node that links to every Fortran file in the graph.
ROOT_NODE_NAME = os.getenv("ROOT_NODE_NAME", "E3SM")

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
# "lmstudio" -> LM Studio's OpenAI-compatible server
# "ollama"   -> Ollama's native /api/chat endpoint
LLM_BACKEND = os.getenv("LLM_BACKEND", "lmstudio").lower()

# Base URLs (default ports: LM Studio configured to 1235 here, Ollama 11434).
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1235/v1/chat/completions")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

# Model name. LM Studio ignores it for the loaded model; Ollama needs a real
# tag that you have pulled (e.g. `ollama pull llama3.1`).
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "10000"))
# Seconds to wait for one reply (large Fortran files can be slow locally).
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))

SYSTEM_PROMPT = (
    "You are an AI that extracts variables, constants, and equations from Fortran code."
)

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Put your Neo4j password here")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chat_with_llm(user_message: str) -> str:
    """Send one prompt to the configured LLM backend and return the reply text.

    Raises RuntimeError on connection/HTTP/format problems so callers can skip
    the file instead of writing an error message into the output as if it were data.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if LLM_BACKEND == "ollama":
        url = OLLAMA_URL
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,  # one JSON reply instead of a token stream
            "options": {"num_predict": LLM_MAX_TOKENS},
        }
    elif LLM_BACKEND == "lmstudio":
        url = LMSTUDIO_URL
        payload = {"model": LLM_MODEL, "messages": messages, "max_tokens": LLM_MAX_TOKENS}
    else:
        raise RuntimeError(f"Unknown LLM_BACKEND '{LLM_BACKEND}' (use 'lmstudio' or 'ollama')")

    try:
        response = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if LLM_BACKEND == "ollama":
            return data["message"]["content"]
        return data["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"LLM request to {url} failed: {e}") from e


def get_neo4j_driver():
    """Return a connected Neo4j driver (caller must close it)."""
    from neo4j import GraphDatabase  # imported lazily: only step 3 needs it

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()  # fail early with a clear error
    return driver
