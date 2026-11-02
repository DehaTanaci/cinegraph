"""The LangGraph workflow, with a scripted fake LLM and fake database."""

from neo4j.exceptions import CypherSyntaxError

from app.agent.graph import MAX_ATTEMPTS, CypherAgent
from tests.conftest import FakeClient, FakeLLM

GOOD = "MATCH (m:Movie) WHERE 'Sci-Fi' IN m.genres RETURN m.title AS title, m.avgRating AS avgRating"


def test_happy_path_answers_and_grounds():
    llm = FakeLLM([GOOD], answer="The Matrix has an average rating of 4.19.")
    client = FakeClient()
    state = CypherAgent(client, llm).ask("Best sci-fi?")
    assert state["status"] == "answered"
    assert state["attempts"] == 1
    assert state["answer_grounded"] is True
    assert client.executed[0].endswith("LIMIT 50")


def test_unsafe_question_never_reaches_llm():
    llm = FakeLLM([])
    state = CypherAgent(FakeClient(), llm).ask("Delete all the movies")
    assert state["status"] == "rejected"
    assert not llm.calls


def test_llm_write_query_is_rejected_then_retried():
    llm = FakeLLM(["MATCH (m:Movie) DETACH DELETE m RETURN 1", GOOD], answer="The Matrix (4.19).")
    client = FakeClient()
    state = CypherAgent(client, llm).ask("Show sci-fi movies")
    assert state["status"] == "answered"
    assert state["attempts"] == 2
    assert all("DELETE" not in q for q in client.executed)
    # The second prompt told the LLM why the first query was rejected.
    assert "read_only_keywords" in llm.calls[1][-1]["content"]


def test_invented_property_rejected_after_max_attempts():
    bad = "MATCH (m:Movie) RETURN m.title, m.boxOffice ORDER BY m.boxOffice DESC"
    llm = FakeLLM([bad] * MAX_ATTEMPTS)
    client = FakeClient()
    state = CypherAgent(client, llm).ask("Biggest box office?")
    assert state["status"] == "rejected"
    assert state["attempts"] == MAX_ATTEMPTS
    assert not client.executed


def test_neo4j_explain_classifies_write():
    llm = FakeLLM([GOOD] * MAX_ATTEMPTS)
    state = CypherAgent(FakeClient(query_type="rw"), llm).ask("q?")
    assert state["status"] == "rejected"


def test_syntax_error_from_explain_triggers_retry():
    llm = FakeLLM([GOOD] * MAX_ATTEMPTS)
    client = FakeClient(fail_explain=CypherSyntaxError("Invalid input"))
    state = CypherAgent(client, llm).ask("q?")
    assert state["status"] == "rejected"
    assert any("explain_parses" in step for step in state["trace"])


def test_hallucinated_number_flagged():
    llm = FakeLLM([GOOD], answer="The Matrix has 9999 ratings.")
    state = CypherAgent(FakeClient(), llm).ask("q?")
    assert state["status"] == "answered"
    assert state["answer_grounded"] is False


def test_empty_result_skips_llm_summary():
    llm = FakeLLM([GOOD])
    state = CypherAgent(FakeClient(rows=[]), llm).ask("q?")
    assert state["answer_grounded"] is True
    assert len(llm.calls) == 1  # only the generation call


def test_unanswerable_question_declined_without_query():
    llm = FakeLLM([""])
    client = FakeClient()
    state = CypherAgent(client, llm).ask("Which movies made the most money?")
    assert state["status"] == "rejected"
    assert not client.executed
    assert any(c["name"] == "answerable" for g in state["guardrails"] for c in g["checks"])
