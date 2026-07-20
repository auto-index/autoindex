"""
Shared fixtures for analysis_code_agent tests.
"""
import subprocess
import sys
import time
import pathlib
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
_EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"

# Ensure evaluation schema is importable
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from schema import Document, EvalQuery, Chunk  # noqa: E402

# ── Minimal synthetic corpus ─────────────────────────────────────────────────
# Two Wikipedia "articles", each with a title stub (:0) and content section (:1).
# Article 11111 = "The Matrix" (sci-fi film)
# Article 22222 = "Inception" (mind-bending thriller)

SAMPLE_DOCS = [
    Document(doc_id="11111:0", text="# The Matrix\n1999 science fiction film", metadata={}),
    Document(doc_id="11111:1", text="# The Matrix\nThe Matrix is a 1999 American science fiction action film "
             "written and directed by the Wachowskis. It stars Keanu Reeves as Neo, "
             "a hacker who discovers the world is a simulation.", metadata={}),
    Document(doc_id="22222:0", text="# Inception\n2010 science fiction film", metadata={}),
    Document(doc_id="22222:1", text="# Inception\nInception is a 2010 science fiction action film "
             "written and directed by Christopher Nolan. A thief steals information "
             "from within the subconscious mind during the dream state.", metadata={}),
]

# Queries where the gold doc is the content section (:1)
SAMPLE_QUERIES = [
    EvalQuery(
        query_id="q_matrix",
        query_text="1999 sci-fi film about a hacker who discovers reality is a simulation",
        relevant_doc_ids=["11111:1"],
    ),
    EvalQuery(
        query_id="q_inception",
        query_text="Christopher Nolan film about stealing secrets through dreams",
        relevant_doc_ids=["22222:1"],
    ),
]

# ── BM25 server fixture (integration tests only) ─────────────────────────────

TEST_SERVER_PORT = 8766  # separate port to avoid conflicts with dev server on 8765


@pytest.fixture(scope="session")
def bm25_server():
    """Start a BM25 server on TEST_SERVER_PORT for the test session."""
    server_path = _PROJECT_ROOT / "src" / "agents" / "analysis_code_agent" / "bm25_server.py"
    proc = subprocess.Popen(
        ["uv", "run", "python", str(server_path), "--port", str(TEST_SERVER_PORT)],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait up to 10s for server to be ready
    from src.agents.analysis_code_agent.bm25_client import BM25Client
    client = BM25Client(base_url=f"http://localhost:{TEST_SERVER_PORT}")
    for _ in range(20):
        time.sleep(0.5)
        if client.health():
            break
    else:
        proc.terminate()
        pytest.fail("BM25 server failed to start within 10s")

    yield client

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def bm25_client_with_current_index(bm25_server):
    """Build a 'current' index using the baseline (passthrough) preprocessor on SAMPLE_DOCS."""
    client = bm25_server
    # Pre-delete in case a previous test left it behind
    try:
        client.delete_index("current")
    except Exception:
        pass
    # Passthrough: one chunk per doc, text unchanged
    chunks = [
        Chunk(chunk_id=f"{d.doc_id}_0", doc_id=d.doc_id, text=d.text)
        for d in SAMPLE_DOCS
    ]
    client.build_index("current", chunks, persist=False)
    yield client
    # Clean up current index after each test that uses it
    try:
        client.delete_index("current")
    except Exception:
        pass
