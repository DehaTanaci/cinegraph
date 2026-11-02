"""Integration tests against the real database: connectivity, schema, and the Cypher queries' behaviour.

Skipped automatically if Neo4j is not running or empty.
"""

import pytest

from app.agent.graph import CypherAgent
from app.services.gds_service import GdsService
from app.services.movie_service import MovieService, NotFoundError
from tests.conftest import FakeLLM

pytestmark = pytest.mark.integration


@pytest.fixture(name="service")
def service_fixture(neo4j_client):
    """Real movie service."""
    return MovieService(neo4j_client)


def test_connectivity(neo4j_client):
    assert neo4j_client.verify()
    assert neo4j_client.read("RETURN 1 AS ok") == [{"ok": 1}]


def test_constraints_and_indexes_exist(neo4j_client):
    names = {r["name"] for r in neo4j_client.read("SHOW INDEXES YIELD name RETURN name")}
    assert {"movie_id", "user_id", "genre_name", "tag_name", "movie_title_ft", "movie_year", "rated_timestamp"} <= names


def test_movie_id_is_unique(neo4j_client):
    rows = neo4j_client.read("MATCH (m:Movie) WITH m.movieId AS id, count(*) AS n WHERE n > 1 RETURN count(*) AS d")
    assert rows[0]["d"] == 0


def test_lookup_uses_index_seek(neo4j_client):
    plan = neo4j_client.explain("MATCH (m:Movie {movieId: 1}) RETURN m")
    assert any("NodeUniqueIndexSeek" in op for op in plan["operators"])


def test_genres_property_matches_genre_nodes(neo4j_client):
    """The denormalised `genres` list must agree with the IN_GENRE relationships."""
    rows = neo4j_client.read(
        """
        MATCH (m:Movie)
        WITH m, [(m)-[:IN_GENRE]->(g) | g.name] AS viaNodes
        WHERE size(viaNodes) <> size(m.genres) OR any(x IN m.genres WHERE NOT x IN viaNodes)
        RETURN count(m) AS mismatches
        """
    )
    assert rows[0]["mismatches"] == 0


def test_aggregates_are_consistent(neo4j_client):
    rows = neo4j_client.read(
        "MATCH (m:Movie {movieId: 1}) RETURN m.ratingCount AS stored, COUNT { (m)<-[:RATED]-() } AS live"
    )
    assert rows[0]["stored"] == rows[0]["live"]


def test_recommendations_exclude_already_rated(service, neo4j_client):
    recs = service.recommendations(1, limit=20, offset=0)
    assert recs
    rated = {r["id"] for r in neo4j_client.read(
        "MATCH (:User {userId: 1})-[:RATED]->(m) RETURN m.movieId AS id")}
    assert not {r["movie"]["movie_id"] for r in recs} & rated
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_genre_filter(service):
    recs = service.recommendations(1, limit=10, offset=0, genre="Horror")
    assert all("Horror" in r["movie"]["genres"] for r in recs)


def test_pagination_is_disjoint(service):
    first = [r["movie"]["movie_id"] for r in service.trending("rating", 0, 50, 5, 0)]
    second = [r["movie"]["movie_id"] for r in service.trending("rating", 0, 50, 5, 5)]
    assert first and second and not set(first) & set(second)


def test_content_similarity_shares_features(service):
    rows = service.similar_movies(1, "content", limit=5, offset=0)
    assert rows and all(0 < r["score"] <= 1 for r in rows)


def test_unknown_ids_raise(service):
    with pytest.raises(NotFoundError):
        service.get_movie(-1)
    with pytest.raises(NotFoundError):
        service.recommendations(-1, 5, 0)


def test_search_finds_title(service):
    assert service.search_movies("toy story", 3, 0)[0]["movie"]["title"].startswith("Toy Story")


def test_user_profile(service):
    profile = service.get_user(1)
    assert profile["rating_count"] > 0 and profile["favourite_genres"]


def test_trending_recent(service):
    rows = service.trending("recent", 365, 5, 5, 0)
    assert rows and rows[0]["score"] >= rows[-1]["score"]


@pytest.fixture(name="with_gds")
def gds_fixture(neo4j_client):
    """Skip GDS-dependent tests if algorithms have not been run."""
    status = GdsService(neo4j_client).status()
    if not status["similarRelationships"]:
        pytest.skip("Run `make gds` first")
    return status


def test_gds_outputs_present(with_gds):
    assert with_gds["moviesWithPagerank"] > 0 and with_gds["usersWithCommunity"] > 0


def test_similarity_scores_in_range(neo4j_client, with_gds):  # pylint: disable=unused-argument
    rows = neo4j_client.read("MATCH ()-[s:SIMILAR]->() RETURN min(s.score) AS lo, max(s.score) AS hi")
    assert 0.1 <= rows[0]["lo"] <= rows[0]["hi"] <= 1.0


def test_collaborative_similar(service, with_gds):  # pylint: disable=unused-argument
    titles = [r["movie"]["title"] for r in service.similar_movies(1, "collaborative", 10, 0)]
    assert "Toy Story 2" in titles


def test_why_recommended_has_evidence(service, with_gds):  # pylint: disable=unused-argument
    rec = service.recommendations(1, 1, 0)[0]["movie"]["movie_id"]
    why = service.why_recommended(1, rec)
    assert why["rated"] is None and (why["peers"] or why["similar"])


def test_discover_outside_home(service, with_gds):  # pylint: disable=unused-argument
    assert service.discover(1, 5, 0)


def test_taste_path_both_methods(service, with_gds):  # pylint: disable=unused-argument
    hops = service.taste_path(1, 858, weighted=False)
    weighted = service.taste_path(1, 858, weighted=True)
    assert hops["steps"][0]["movie_id"] == 1 and hops["steps"][-1]["movie_id"] == 858
    assert weighted["total_cost"] >= 0 and weighted["hops"] >= hops["hops"]


def test_agent_against_real_db(neo4j_client):
    """Guardrails + real EXPLAIN + real read transaction, with a scripted LLM."""
    llm = FakeLLM(["MATCH (m:Movie) WHERE m.ratingCount >= 100 RETURN m.title AS title, m.avgRating AS r "
                   "ORDER BY r DESC LIMIT 3"], answer="See the list.")
    state = CypherAgent(neo4j_client, llm).ask("Top movies?")
    assert state["status"] == "answered" and len(state["rows"]) == 3


def test_read_transaction_blocks_writes(neo4j_client):
    """Even if every check were bypassed, a read transaction cannot write."""
    with pytest.raises(Exception):
        neo4j_client.read("CREATE (x:Movie {movieId: -42}) RETURN x")
    assert not neo4j_client.read("MATCH (m:Movie {movieId: -42}) RETURN m")
