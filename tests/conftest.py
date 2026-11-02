"""Shared fixtures. Integration tests are skipped automatically when Neo4j is not running."""

import pytest

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient


@pytest.fixture(scope="session")
def neo4j_client():
    """A real client against the configured database, or skip."""
    settings = get_settings()
    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    if not client.verify():
        client.close()
        pytest.skip("Neo4j is not reachable (start it with `make docker-run`)")
    if not client.read("MATCH (m:Movie) RETURN m LIMIT 1"):
        client.close()
        pytest.skip("Neo4j is empty (load data with `make seed`)")
    yield client
    client.close()


class FakeLLM:
    """Returns scripted responses in order; records the prompts it saw."""

    def __init__(self, cypher_replies=None, answer="ok"):
        self.cypher_replies = list(cypher_replies or [])
        self.answer = answer
        self.calls = []

    def chat_json(self, messages, schema):  # pylint: disable=unused-argument
        """Next scripted query."""
        self.calls.append(messages)
        return {"cypher": self.cypher_replies.pop(0), "reasoning": "test"}

    def chat(self, messages, schema=None):  # pylint: disable=unused-argument
        """Scripted answer text."""
        self.calls.append(messages)
        return self.answer

    def is_available(self):
        """Always available."""
        return True


class FakeClient:
    """Stands in for Neo4jClient: EXPLAIN says read-only, read returns fixed rows."""

    def __init__(self, rows=None, query_type="r", operators=None, fail_explain=None):
        self.rows = rows if rows is not None else [{"title": "The Matrix", "avgRating": 4.19}]
        self.query_type = query_type
        self.operators = operators or ["ProduceResults@neo4j", "NodeByLabelScan@neo4j"]
        self.fail_explain = fail_explain
        self.executed = []

    def explain(self, cypher, params=None):  # pylint: disable=unused-argument
        """Pretend plan."""
        if self.fail_explain:
            raise self.fail_explain
        return {"query_type": self.query_type, "operators": self.operators}

    def read(self, cypher, params=None, timeout=None):  # pylint: disable=unused-argument
        """Pretend rows."""
        self.executed.append(cypher)
        return self.rows

    def verify(self):
        """Always reachable."""
        return True
