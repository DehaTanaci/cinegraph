"""Safety checks applied to the question and to every LLM-generated Cypher query.

Defence in depth — each layer catches what the previous one might miss:
1. question screening   - refuse requests to modify data or override instructions;
2. static query checks  - one statement, no write/admin keywords, only known labels/relationships/properties;
3. row cap              - LIMIT forced to at most `max_rows`;
4. EXPLAIN              - Neo4j itself confirms the query parses and is read-only ('r');
5. execution            - read-only transaction + timeout (the server refuses writes even if 1-4 failed);
6. answer grounding     - numbers in the AI's answer must exist in the returned rows.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from app.database.schema import LLM_SCHEMA

WRITE_INTENT = re.compile(
    r"\b(delete|remove|drop|erase|wipe|insert|create|add|update|modify|change|rename|merge|set)\b"
    r".{0,40}\b(movie|movies|user|users|rating|ratings|node|nodes|database|graph|genre|tag|data|everything)\b",
    re.IGNORECASE,
)
INJECTION = re.compile(
    r"ignore (all |the |your )?(previous |prior |above )?(instructions|rules)|system prompt|you are now|"
    r"disregard (the |your )?(rules|instructions)|jailbreak",
    re.IGNORECASE,
)
FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|ALTER|GRANT|DENY|REVOKE|"
    r"TERMINATE|IMPORT|USING\s+PERIODIC\s+COMMIT|SHOW)\b"
    r"|\bCALL\s+(dbms|db|apoc|gds|tx)\b|\bIN\s+TRANSACTIONS\b",
    re.IGNORECASE,
)
STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
LABEL = re.compile(r"\(\s*\w*\s*((?::\s*`?\w+`?\s*)+)")
REL_TYPE = re.compile(r"\[\s*\w*\s*:\s*([`\w|:!&]+)")
PROPERTY = re.compile(r"\b[a-zA-Z_]\w*\.([a-zA-Z_]\w*)\b")
LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")

ALLOWED_LABELS = set(LLM_SCHEMA["nodes"])
ALLOWED_RELS = set(LLM_SCHEMA["relationships"])
ALLOWED_PROPS = {p for props in LLM_SCHEMA["nodes"].values() for p in props} | {
    p for spec in LLM_SCHEMA["relationships"].values() for p in spec["properties"]
}
# Temporal accessors that are valid on `timestamp`, e.g. r.timestamp.year
ALLOWED_PROPS |= {"year", "month", "day", "epochSeconds", "epochMillis"}


@dataclass
class CheckResult:
    """Outcome of one guardrail."""

    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialise for API responses."""
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ValidationOutcome:
    """Aggregate of all static checks, plus the (possibly rewritten) query."""

    cypher: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if every check passed."""
        return all(c.passed for c in self.checks)

    def errors(self) -> list[str]:
        """Human-readable failures, fed back to the LLM on retry."""
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]


def screen_question(question: str) -> CheckResult:
    """Refuse questions that ask to change data or try to override the assistant's rules."""
    if INJECTION.search(question):
        return CheckResult("question_screen", False, "Question tries to override the assistant's instructions.")
    if WRITE_INTENT.search(question):
        return CheckResult("question_screen", False, "Only read-only questions are allowed; this asks to modify data.")
    return CheckResult("question_screen", True, "read-only question")


def strip_strings(cypher: str) -> str:
    """Replace string literals with '' so a title like 'Set It Up' does not look like a SET clause."""
    return STRING_LITERAL.sub("''", cypher)


def normalise(cypher: str) -> str:
    """Trim whitespace, code fences and a single trailing semicolon."""
    text = cypher.strip()
    text = re.sub(r"^```(?:cypher)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text[:-1].rstrip() if text.endswith(";") else text


def check_single_statement(cypher: str) -> CheckResult:
    """Exactly one statement: no ';' outside string literals."""
    ok = ";" not in strip_strings(cypher)
    return CheckResult("single_statement", ok, "" if ok else "Multiple statements are not allowed.")


def check_read_only(cypher: str) -> CheckResult:
    """No write, schema, admin or procedure-call keywords."""
    match = FORBIDDEN.search(strip_strings(cypher))
    if match:
        return CheckResult("read_only_keywords", False, f"Forbidden keyword: {match.group(0).strip()}")
    return CheckResult("read_only_keywords", True)


def check_has_return(cypher: str) -> CheckResult:
    """A query without RETURN produces nothing to show."""
    ok = re.search(r"\bRETURN\b", strip_strings(cypher), re.IGNORECASE) is not None
    return CheckResult("has_return", ok, "" if ok else "Query must RETURN something.")


def check_schema(cypher: str) -> CheckResult:
    """Every label, relationship type and property must exist in the schema the LLM was given."""
    text = strip_strings(cypher)
    labels = {lab.strip(" `") for group in LABEL.findall(text) for lab in group.split(":") if lab.strip(" `")}
    rels = {r.strip(" `!&") for group in REL_TYPE.findall(text) for r in re.split(r"[|:]", group) if r.strip(" `!&")}
    props = set(PROPERTY.findall(text))
    problems = []
    if labels - ALLOWED_LABELS:
        problems.append(f"unknown labels {sorted(labels - ALLOWED_LABELS)}")
    if rels - ALLOWED_RELS:
        problems.append(f"unknown relationship types {sorted(rels - ALLOWED_RELS)}")
    if props - ALLOWED_PROPS:
        problems.append(f"unknown properties {sorted(props - ALLOWED_PROPS)}")
    if problems:
        return CheckResult("schema_allowlist", False, "; ".join(problems))
    return CheckResult("schema_allowlist", True, f"labels={sorted(labels)} rels={sorted(rels)}")


def enforce_limit(cypher: str, max_rows: int) -> tuple[str, CheckResult]:
    """Append LIMIT if missing, lower it if above `max_rows`."""
    match = LIMIT.search(cypher)
    if match is None:
        return f"{cypher}\nLIMIT {max_rows}", CheckResult("row_limit", True, f"added LIMIT {max_rows}")
    if int(match.group(1)) > max_rows:
        return LIMIT.sub(f"LIMIT {max_rows}", cypher), CheckResult("row_limit", True, f"lowered to {max_rows}")
    return cypher, CheckResult("row_limit", True, f"LIMIT {match.group(1)}")


def validate_cypher(cypher: str, max_rows: int) -> ValidationOutcome:
    """Run all static checks. If they pass, the returned query has an enforced LIMIT."""
    text = normalise(cypher)
    outcome = ValidationOutcome(text)
    outcome.checks += [check_single_statement(text), check_read_only(text), check_has_return(text), check_schema(text)]
    if outcome.passed:
        outcome.cypher, limit_check = enforce_limit(text, max_rows)
        outcome.checks.append(limit_check)
    return outcome


def check_explain(plan: dict[str, Any]) -> list[CheckResult]:
    """Interpret Neo4j's EXPLAIN output: must be read-only; warn on full scans / Cartesian products."""
    results = [
        CheckResult(
            "explain_read_only", plan["query_type"] == "r", f"Neo4j classified the query as '{plan['query_type']}'"
        )
    ]
    operators = [op.split("@")[0] for op in plan["operators"]]
    # A Cartesian product of two index seeks (e.g. two movies by id) is cheap; one fed by a full
    # label scan multiplies the size of the graph by itself and is what we refuse.
    if "CartesianProduct" in operators and any(op in ("AllNodesScan", "NodeByLabelScan") for op in operators):
        results.append(CheckResult("explain_no_cartesian", False, "Plan contains a Cartesian product over a full "
                                                                  "scan (unconnected patterns): connect them."))
    return results


def _numbers(value: Any) -> set[float]:
    found: set[float] = set()
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.add(round(float(value), 2))
    elif isinstance(value, dict):
        for item in value.values():
            found |= _numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= _numbers(item)
    elif isinstance(value, str):
        found |= {round(float(n), 2) for n in NUMBER.findall(value)}
    return found


def check_grounding(answer: str, rows: list[dict[str, Any]], question: str) -> CheckResult:
    """Every number in the answer must appear in the data (or in the question itself).

    Catches the most common hallucination: invented counts, ratings and years.
    Numbers are compared after rounding to 2 decimals, so 4.4333 in the data matches "4.43".
    """
    allowed = _numbers(rows) | _numbers(question)
    allowed |= {round(n, 1) for n in allowed} | {float(round(n)) for n in allowed}
    allowed |= {float(len(rows))} | {float(i) for i in range(1, 11)}  # list positions "the 3 movies", "1."
    claimed = {round(float(n), 2) for n in NUMBER.findall(answer)}
    unsupported = sorted(claimed - allowed)
    if unsupported:
        return CheckResult("answer_grounding", False, f"numbers not found in the data: {unsupported}")
    return CheckResult("answer_grounding", True, f"{len(claimed)} numbers checked")
