"""API tests with fake services: validation, error codes, auth, response shapes."""

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from app import dependencies
from app.agent.graph import CypherAgent
from app.config import Settings, get_settings
from app.database.neo4j_client import get_client
from app.main import app
from app.services.movie_service import NotFoundError, PrerequisiteError
from tests.conftest import FakeClient, FakeLLM

MOVIE = {"movie_id": 1, "title": "Toy Story", "year": 1995, "genres": ["Animation"], "avg_rating": 3.9,
         "rating_count": 215}


class FakeMovieService:
    """Returns canned rows; raises for id 404/409/503 to exercise error handlers."""

    @staticmethod
    def _maybe_fail(some_id):
        if some_id == 404:
            raise NotFoundError("Movie 404 not found")
        if some_id == 409:
            raise PrerequisiteError("Run `make gds`")
        if some_id == 503:
            raise ServiceUnavailable("down")

    def get_movie(self, movie_id):
        """One movie."""
        self._maybe_fail(movie_id)
        return {**MOVIE, "top_tags": ["pixar"]}

    def search_movies(self, text, limit, offset):  # pylint: disable=unused-argument
        """Search."""
        return [{"movie": MOVIE, "score": 1.0}]

    def similar_movies(self, movie_id, method, limit, offset):  # pylint: disable=unused-argument
        """Similar."""
        self._maybe_fail(movie_id)
        return [{"movie": MOVIE, "score": 0.5, "detail": method}]

    def trending(self, metric, window_days, min_ratings, limit, offset):  # pylint: disable=unused-argument
        """Ranked."""
        return [{"movie": MOVIE, "score": 2.0}]

    def taste_path(self, from_id, to_id, weighted):
        """Path."""
        return {"method": "dijkstra" if weighted else "shortestPath", "hops": 1,
                "steps": [{"movie_id": from_id, "title": "A", "similarity_to_next": 0.3},
                          {"movie_id": to_id, "title": "B"}]}

    def get_user(self, user_id):
        """Profile."""
        self._maybe_fail(user_id)
        return {"user_id": user_id, "rating_count": 3, "avg_rating": 4.0, "favourite_genres": [], "community": 1}

    def recommendations(self, user_id, limit, offset, genre=None):  # pylint: disable=unused-argument
        """Recs."""
        self._maybe_fail(user_id)
        return [{"movie": MOVIE, "score": 10.0, "detail": "3 users"}]

    def discover(self, user_id, limit, offset):  # pylint: disable=unused-argument
        """Discover."""
        self._maybe_fail(user_id)
        return [{"movie": MOVIE, "score": 4.0}]

    def why_recommended(self, user_id, movie_id):
        """Evidence."""
        self._maybe_fail(movie_id)
        return {
            "rated": None, "movie": MOVIE,
            "peers": [{"peer": 7, "overlap": 12, "examples": ["Aladdin"], "peerRating": 5.0}],
            "similar": [{"liked": "Toy Story 2", "rating": 5.0, "score": 0.3}],
            "genres": [{"genre": "Animation", "likedInGenre": 4}],
            "tags": [{"tag": "pixar", "via": ["Up"], "n": 1}],
        } if user_id != 2 else {"rated": 4.5, "movie": MOVIE, "peers": [], "similar": [], "genres": [], "tags": []}


class FakePrediction:
    """Prediction stub."""

    def predict(self, user_id, movie_id):
        """Fixed probability."""
        if movie_id == 409:
            raise PrerequisiteError("No trained model")
        return {"user_id": user_id, "movie_id": movie_id, "probability": 0.8, "model": "fake"}


@pytest.fixture(name="api")
def api_fixture():
    """Test client with all external dependencies replaced."""
    app.dependency_overrides[dependencies.movie_service] = FakeMovieService
    app.dependency_overrides[dependencies.prediction_service] = FakePrediction
    app.dependency_overrides[get_client] = FakeClient
    app.dependency_overrides[get_settings] = lambda: Settings(admin_api_key="secret", _env_file=None)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_docs_available(api):
    assert api.get("/docs").status_code == 200
    assert "/api/ask" in api.get("/openapi.json").json()["paths"]


def test_movie_ok_and_not_found(api):
    assert api.get("/api/movies/1").json()["top_tags"] == ["pixar"]
    response = api.get("/api/movies/404")
    assert response.status_code == 404 and "not found" in response.json()["detail"]


def test_prerequisite_returns_409(api):
    assert api.get("/api/movies/409/similar").status_code == 409


def test_database_down_returns_503(api):
    assert api.get("/api/movies/503").status_code == 503


@pytest.mark.parametrize("url", ["/api/movies/abc", "/api/movies/1/similar?limit=0", "/api/movies/1/similar?limit=51",
                                 "/api/movies/1/similar?method=magic", "/api/movies/search?q=a",
                                 "/api/movies/trending?metric=hype"])
def test_input_validation_422(api, url):
    assert api.get(url).status_code == 422


def test_lists_are_paginated(api):
    body = api.get("/api/movies/1/similar?limit=5&offset=10").json()
    assert body["page"] == {"limit": 5, "offset": 10, "count": 1}
    assert api.get("/api/movies/trending?metric=pagerank").json()["method"] == "pagerank"
    assert api.get("/api/movies/search?q=toy").json()[0]["title"] == "Toy Story"


def test_path(api):
    body = api.get("/api/movies/path?from=1&to=2&weighted=true").json()
    assert body["method"] == "dijkstra" and len(body["steps"]) == 2


def test_user_endpoints(api):
    assert api.get("/api/users/1").json()["user_id"] == 1
    assert api.get("/api/users/1/recommendations?genre=Sci-Fi").json()["items"][0]["score"] == 10.0
    assert api.get("/api/users/1/discover").json()["method"] == "louvain_community"
    assert api.get("/api/users/404").status_code == 404
    assert api.get("/api/users/1/predict/5").json()["probability"] == 0.8
    assert api.get("/api/users/1/predict/409").status_code == 409


def test_why_uses_graph_evidence_only(api):
    body = api.get("/api/users/1/recommendations/1/why").json()
    assert body["source"] == "graph"
    assert {e["kind"] for e in body["evidence"]} == {"similar_users", "similar_movie", "genre_affinity",
                                                     "shared_tag"}
    assert body["summary"].startswith("Recommended because")
    assert api.get("/api/users/2/recommendations/1/why").json()["already_rated"] is True


def test_admin_requires_key(api):
    assert api.get("/api/admin/gds/status").status_code == 401
    assert api.get("/api/admin/gds/status", headers={"X-API-Key": "wrong"}).status_code == 401


def test_health(api, monkeypatch):
    monkeypatch.setattr("app.routers.admin.get_llm", FakeLLM)
    assert api.get("/health").json() == {"status": "ok", "neo4j": True, "llm": True}


def test_ask_endpoint(api):
    llm = FakeLLM(["MATCH (m:Movie) RETURN m.title AS title"], answer="The Matrix (4.19)")
    app.dependency_overrides[dependencies.cypher_agent] = lambda: CypherAgent(FakeClient(), llm)
    body = api.post("/api/ask", json={"question": "Which movies?"}).json()
    assert body["status"] == "answered" and body["rows"] and body["answer_grounded"] is True
    assert api.post("/api/ask", json={"question": "Delete all movies"}).json()["status"] == "rejected"
    assert api.post("/api/ask", json={"question": "x"}).status_code == 422


def test_guardrail_check_endpoint(api):
    blocked = api.post("/api/ask/check", json={"cypher": "MATCH (m:Movie) DETACH DELETE m"}).json()
    assert blocked["passed"] is False
    assert any(c["name"] == "read_only_keywords" and not c["passed"] for c in blocked["checks"])
    allowed = api.post("/api/ask/check", json={"cypher": "MATCH (m:Movie) RETURN m.title"}).json()
    assert allowed["passed"] is True


def test_demo_page_served(api):
    assert api.get("/", follow_redirects=False).headers["location"] == "/demo"
    page = api.get("/demo")
    assert page.status_code == 200 and "CineGraph" in page.text
