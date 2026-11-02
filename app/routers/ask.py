"""Natural-language question endpoint (NL-to-Cypher with guardrails)."""

from fastapi import APIRouter, Depends
from neo4j.exceptions import Neo4jError

from app.agent import guardrails
from app.agent.graph import CypherAgent
from app.config import Settings, get_settings
from app.database.neo4j_client import Neo4jClient, get_client
from app.dependencies import cypher_agent
from app.models.schemas import AskRequest, AskResponse, CypherCheckRequest, GuardrailReport

router = APIRouter(prefix="/api", tags=["assistant"])


@router.post("/ask", response_model=AskResponse, summary="Ask a question in plain English")
def ask(body: AskRequest, agent: CypherAgent = Depends(cypher_agent)):
    """Translate the question to Cypher with the local LLM, validate it, run it read-only, and summarise.

    `rows` are verified database results. `answer` is AI-written and checked: `answer_grounded` is False
    if it contains numbers that are not in `rows`. `guardrails` lists every safety check and its outcome.
    """
    state = agent.ask(body.question)
    return AskResponse(
        question=body.question,
        status=state["status"],
        cypher=state.get("cypher"),
        attempts=state["attempts"],
        guardrails=state.get("guardrails", []),
        rows=state["rows"],
        answer=state.get("answer"),
        answer_grounded=state.get("answer_grounded"),
        trace=state.get("trace", []),
    )


@router.post("/ask/check", response_model=GuardrailReport, summary="Run the guardrails on a Cypher query")
def check_cypher(
    body: CypherCheckRequest,
    client: Neo4jClient = Depends(get_client),
    settings: Settings = Depends(get_settings),
):
    """Apply exactly the checks an LLM-generated query goes through, without running it.

    Use it to demonstrate what happens when the model produces a dangerous or invalid query,
    e.g. `MATCH (m:Movie) DETACH DELETE m` or a query using a property that does not exist.
    """
    outcome = guardrails.validate_cypher(body.cypher, settings.llm_max_rows)
    checks = list(outcome.checks)
    if outcome.passed:
        try:
            checks += guardrails.check_explain(client.explain(outcome.cypher))
        except Neo4jError as exc:
            checks.append(guardrails.CheckResult("explain_parses", False, exc.message or str(exc)))
    return GuardrailReport(passed=all(c.passed for c in checks), checks=[c.as_dict() for c in checks])
