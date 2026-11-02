"""LangGraph workflow for the NL-to-Cypher assistant.

    screen ──(unsafe question)──────────────────────► END (rejected)
       │
       ▼
    generate ──(LLM: data not in graph)─────────────► END (rejected)
       │
       ▼
    validate ──(failed, retries left)───────────────► generate
       │  └────(failed, no retries left)────────────► END (rejected)
       ▼
    execute ──(database error, retries left)────────► generate
       │
       ▼
    summarize ──(grounding check)───────────────────► END (answered)

The LLM only ever proposes text. Deciding whether the query runs, and what counts as a fact,
is done by deterministic code and by Neo4j.
"""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from neo4j.exceptions import Neo4jError

from app.agent import guardrails
from app.agent.llm import LLMError, OllamaClient
from app.agent.prompts import CYPHER_SCHEMA, answer_messages, cypher_messages
from app.database.neo4j_client import Neo4jClient

MAX_ATTEMPTS = 3
QUERY_TIMEOUT_SECONDS = 10.0


def _jsonable(value: Any) -> Any:
    """Convert Neo4j values (nodes, dates, ...) to plain JSON types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "items"):  # Node / Relationship
        return {k: _jsonable(v) for k, v in dict(value.items()).items() if k != "embedding"}
    return str(value)


class AskState(TypedDict, total=False):
    """Everything the workflow knows while answering one question."""

    question: str
    cypher: Optional[str]
    attempts: int
    errors: list[str]
    guardrails: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    answer: Optional[str]
    answer_grounded: Optional[bool]
    status: str
    trace: list[str]


class CypherAgent:
    """Builds and runs the LangGraph state machine."""

    def __init__(self, client: Neo4jClient, llm: OllamaClient, max_rows: int = 50):
        self.client = client
        self.llm = llm
        self.max_rows = max_rows
        self.graph = self._build()

    # ----- nodes -----

    def screen(self, state: AskState) -> AskState:
        """Reject unsafe questions before any LLM call."""
        check = guardrails.screen_question(state["question"])
        report = {"passed": check.passed, "checks": [check.as_dict()]}
        trace = state.get("trace", []) + [f"screen: {'ok' if check.passed else check.detail}"]
        if not check.passed:
            return {"status": "rejected", "guardrails": [report], "trace": trace, "errors": [check.detail]}
        return {"guardrails": [report], "trace": trace, "attempts": 0, "errors": []}

    def generate(self, state: AskState) -> AskState:
        """Ask the LLM for a query (with the previous attempt's errors, if any)."""
        attempts = state.get("attempts", 0) + 1
        trace = state["trace"] + [f"generate: attempt {attempts}"]
        try:
            data = self.llm.chat_json(
                cypher_messages(state["question"], state.get("errors", []), state.get("cypher")), CYPHER_SCHEMA
            )
        except LLMError as exc:
            return {"status": "error", "attempts": attempts, "errors": [str(exc)], "trace": trace + [str(exc)]}
        cypher = str(data.get("cypher", "")).strip()
        if not cypher:
            reason = str(data.get("reasoning", "")).strip() or "the graph does not contain the data needed"
            check = guardrails.CheckResult("answerable", False, f"LLM declined: {reason}")
            return {"status": "rejected", "attempts": attempts, "cypher": None, "errors": [check.detail],
                    "guardrails": state["guardrails"] + [{"passed": False, "checks": [check.as_dict()]}],
                    "trace": trace + [f"not answerable: {reason}"]}
        return {"cypher": cypher, "attempts": attempts, "trace": trace + [f"llm proposed: {cypher}"]}

    def validate(self, state: AskState) -> AskState:
        """Static checks, then EXPLAIN on the database."""
        outcome = guardrails.validate_cypher(state.get("cypher") or "", self.max_rows)
        checks = list(outcome.checks)
        if outcome.passed:
            try:
                checks += guardrails.check_explain(self.client.explain(outcome.cypher))
            except Neo4jError as exc:
                checks.append(guardrails.CheckResult("explain_parses", False, exc.message or str(exc)))
        passed = all(c.passed for c in checks)
        errors = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
        report = {"passed": passed, "checks": [c.as_dict() for c in checks]}
        return {
            "cypher": outcome.cypher,
            "errors": errors,
            "guardrails": state["guardrails"] + [report],
            "trace": state["trace"] + [f"validate: {'ok' if passed else '; '.join(errors)}"],
        }

    def execute(self, state: AskState) -> AskState:
        """Run the validated query in a read-only transaction with a timeout."""
        try:
            rows = self.client.read(state["cypher"], timeout=QUERY_TIMEOUT_SECONDS)
        except Neo4jError as exc:
            message = exc.message or str(exc)
            return {"errors": [f"execution: {message}"], "rows": [],
                    "trace": state["trace"] + [f"execute failed: {message}"]}
        rows = [{key: _jsonable(value) for key, value in row.items()} for row in rows]
        return {"rows": rows, "errors": [], "trace": state["trace"] + [f"execute: {len(rows)} rows"]}

    def summarize(self, state: AskState) -> AskState:
        """Summarise rows with the LLM, then verify the summary against the rows."""
        rows = state["rows"]
        if not rows:
            return {"status": "answered", "answer": "The database has no rows matching this question.",
                    "answer_grounded": True, "trace": state["trace"] + ["answer: empty result, no LLM used"]}
        try:
            text = self.llm.chat(answer_messages(state["question"], rows[:30])).strip()
        except LLMError as exc:
            return {"status": "answered", "answer": None, "answer_grounded": None,
                    "trace": state["trace"] + [f"answer: LLM unavailable ({exc}); rows returned as-is"]}
        check = guardrails.check_grounding(text, rows, state["question"])
        report = {"passed": check.passed, "checks": [check.as_dict()]}
        return {
            "status": "answered",
            "answer": text,
            "answer_grounded": check.passed,
            "guardrails": state["guardrails"] + [report],
            "trace": state["trace"] + [f"grounding: {'ok' if check.passed else check.detail}"],
        }

    def reject(self, state: AskState) -> AskState:
        """Give up after too many failed attempts."""
        return {"status": "rejected", "trace": state["trace"] + ["rejected: no safe valid query produced"]}

    # ----- routing -----

    @staticmethod
    def after_screen(state: AskState) -> str:
        """Stop immediately if the question was refused."""
        return END if state.get("status") == "rejected" else "generate"

    @staticmethod
    def after_generate(state: AskState) -> str:
        """Stop if the LLM failed or declared the question unanswerable."""
        return END if state.get("status") in ("error", "rejected") else "validate"

    @staticmethod
    def after_validate(state: AskState) -> str:
        """Run if valid; otherwise retry or reject."""
        if not state["errors"]:
            return "execute"
        return "generate" if state["attempts"] < MAX_ATTEMPTS else "reject"

    @staticmethod
    def after_execute(state: AskState) -> str:
        """Answer if it ran; on database errors retry or reject."""
        if not state["errors"]:
            return "summarize"
        return "generate" if state["attempts"] < MAX_ATTEMPTS else "reject"

    def _build(self):
        graph = StateGraph(AskState)
        for name in ("screen", "generate", "validate", "execute", "summarize", "reject"):
            graph.add_node(name, getattr(self, name))
        graph.set_entry_point("screen")
        graph.add_conditional_edges("screen", self.after_screen)
        graph.add_conditional_edges("generate", self.after_generate)
        graph.add_conditional_edges("validate", self.after_validate)
        graph.add_conditional_edges("execute", self.after_execute)
        graph.add_edge("summarize", END)
        graph.add_edge("reject", END)
        return graph.compile()

    def ask(self, question: str) -> dict[str, Any]:
        """Run the workflow for one question and return the final state."""
        state = self.graph.invoke({"question": question.strip(), "trace": []})
        state.setdefault("attempts", 0)
        state.setdefault("rows", [])
        state.setdefault("status", "error")
        return state
