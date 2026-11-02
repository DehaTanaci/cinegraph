"""Thin wrapper around the official Neo4j driver."""

from typing import Any, Optional

from neo4j import Driver, GraphDatabase, unit_of_work
from neo4j.exceptions import DriverError, Neo4jError

from app.config import get_settings


class Neo4jClient:
    """Owns the driver and exposes read/write helpers that always use parameters."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close the underlying driver and its connection pool."""
        self._driver.close()

    def verify(self) -> bool:
        """Return True if the database is reachable."""
        try:
            self._driver.verify_connectivity()
            return True
        except (DriverError, Neo4jError):
            return False

    def read(self, cypher: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> list[dict[str, Any]]:
        """Run a query in a read-only transaction and return rows as dicts.

        A read transaction is enforced by the server: any write clause fails.
        """

        @unit_of_work(timeout=timeout)
        def work(tx):
            return [record.data() for record in tx.run(cypher, params or {})]

        with self._driver.session() as session:
            return session.execute_read(work)

    def write(self, cypher: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
        """Run a query in a write transaction and return rows as dicts."""

        def work(tx):
            return [record.data() for record in tx.run(cypher, params or {})]

        with self._driver.session() as session:
            return session.execute_write(work)

    def run_auto(self, cypher: str, params: Optional[dict] = None) -> None:
        """Run a query in an auto-commit transaction (needed for CALL ... IN TRANSACTIONS)."""
        with self._driver.session() as session:
            session.run(cypher, params or {}).consume()

    def explain(self, cypher: str, params: Optional[dict] = None) -> dict[str, Any]:
        """Ask Neo4j to plan (not run) a query.

        Returns the query type ('r' read, 'rw', 'w', 's' schema) and the operators in the plan.
        Raises a neo4j error if the query is syntactically invalid or references unknown procedures.
        """
        with self._driver.session() as session:
            result = session.run("EXPLAIN " + cypher, params or {})
            summary = result.consume()
        operators: list[str] = []

        def walk(plan):
            if not plan:
                return
            operators.append(plan.get("operatorType", ""))
            for child in plan.get("children", []):
                walk(child)

        walk(summary.plan)
        return {"query_type": summary.query_type, "operators": operators}

    def profile(self, cypher: str, params: Optional[dict] = None) -> dict[str, Any]:
        """Run a query with PROFILE and return total db hits, rows and server time."""
        with self._driver.session() as session:
            result = session.run("PROFILE " + cypher, params or {})
            rows = len(list(result))
            summary = result.consume()
        hits = 0

        def walk(plan):
            nonlocal hits
            hits += plan.get("dbHits", 0)
            for child in plan.get("children", []):
                walk(child)

        walk(summary.profile)
        server_ms = summary.result_available_after + summary.result_consumed_after
        return {"db_hits": hits, "rows": rows, "server_ms": server_ms}


_CLIENT: Optional[Neo4jClient] = None


def get_client() -> Neo4jClient:
    """Return the process-wide client, creating it on first use."""
    global _CLIENT  # pylint: disable=global-statement
    if _CLIENT is None:
        settings = get_settings()
        _CLIENT = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    return _CLIENT


def close_client() -> None:
    """Close the process-wide client if it exists."""
    global _CLIENT  # pylint: disable=global-statement
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None
