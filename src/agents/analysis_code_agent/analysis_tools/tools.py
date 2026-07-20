"""Structured tools for the AnalysisAgent.

Three tools are exposed via LiteLLM's OpenAI-compatible tool-calling API so the
LLM can investigate BM25 retrieval failures without arbitrary shell access.

When tools change, make sure to update the prompt in src/agents/analysis_code_agent/analysis_agent.py under the build_initial_context() function.
"""

import json
import pathlib
import re

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def bm25_retrieve(client, query: str, top_k: int = 10) -> str:
    """Query the current BM25 index and return results as JSON."""
    try:
        results = client.retrieve(name="current", query=query, top_k=top_k)
        return json.dumps(results)
    except Exception as e:
        return f"Error running bm25_retrieve: {e}"


def _validate_path(split: str, file_path: str) -> pathlib.Path | str:
    """Resolve *file_path* under ``data/<split>/`` and return the path.

    Returns an error string instead of a Path when the request is invalid.
    """
    if file_path == "queries.jsonl" or file_path == "evaluation_queries.jsonl":
        return "Error: Access to evaluation queries is forbidden. Use 'validation_queries.jsonl' instead."

    base_dir = (_PROJECT_ROOT / "data" / split).resolve()
    target = (base_dir / file_path).resolve()

    if not target.is_relative_to(base_dir):
        return f"Error: path '{file_path}' escapes the data directory"

    if target.is_symlink():
        real = target.resolve()
        if not real.is_relative_to(base_dir):
            return f"Error: symlink '{file_path}' points outside the data directory"

    if not target.exists():
        return f"Error: file '{file_path}' not found in data/{split}/"

    return target


def read_file(split: str, file_path: str, max_chars: int = 800, filter_id: str | None = None) -> str:
    """Read file contents from ``data/<split>/``."""
    validated = _validate_path(split, file_path)
    if isinstance(validated, str):
        return validated

    try:
        if filter_id and file_path.endswith(".jsonl"):
            with open(validated, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("doc_id") == filter_id or obj.get("query_id") == filter_id:
                        content = line.strip()
                        if len(content) > max_chars:
                            return content[:max_chars] + "... [truncated]"
                        return content
            return f"Error: no entry with doc_id or query_id '{filter_id}' found in {file_path}"

        content = validated.read_text(encoding="utf-8")
        if len(content) > max_chars:
            return content[:max_chars] + "... [truncated]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def grep_search(split: str, pattern: str, file_path: str, max_results: int = 10) -> str:
    """Search file contents using a regex pattern."""
    validated = _validate_path(split, file_path)
    if isinstance(validated, str):
        return validated

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    matches: list[str] = []
    try:
        with open(validated, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if regex.search(line):
                    matches.append(f"line {line_no}: {line.rstrip()}")
                    if len(matches) >= max_results:
                        break
    except Exception as e:
        return f"Error searching file: {e}"

    if not matches:
        return "No matches found."
    return "\n".join(matches)


# ---------------------------------------------------------------------------
# OpenAI tool schemas (client / split are injected at dispatch time)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bm25_retrieve",
            "description": "Query the current BM25 index to see what documents are retrieved for a given query. Returns doc_id, score, and rank for each result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query text to search"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from the data directory. For JSONL files, use filter_id to return only the entry matching a specific doc_id or query_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to the data directory, e.g. 'documents.jsonl'"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default 800)"},
                    "filter_id": {"type": "string", "description": "For JSONL files: return only the line where doc_id or query_id matches this value"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents using a regex pattern. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex pattern to search for"},
                    "file_path": {"type": "string", "description": "Path relative to the data directory"},
                    "max_results": {"type": "integer", "description": "Maximum matching lines to return (default 10)"},
                },
                "required": ["pattern", "file_path"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def dispatch_tool(name: str, args: dict, client, split: str) -> str:
    """Route a tool call to the appropriate function, injecting client/split."""
    if name == "bm25_retrieve":
        return bm25_retrieve(
            client=client,
            query=args["query"],
            top_k=args.get("top_k", 10),
        )
    elif name == "read_file":
        return read_file(
            split=split,
            file_path=args["file_path"],
            max_chars=args.get("max_chars", 800),
            filter_id=args.get("filter_id"),
        )
    elif name == "grep_search":
        return grep_search(
            split=split,
            pattern=args["pattern"],
            file_path=args["file_path"],
            max_results=args.get("max_results", 10),
        )
    else:
        return f"Error: unknown tool '{name}'"
