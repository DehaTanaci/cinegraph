"""Pydantic models for every request and response the API exposes."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class MovieSummary(BaseModel):
    """A movie as it appears in lists."""

    movie_id: int
    title: str
    year: Optional[int] = None
    genres: list[str] = Field(default_factory=list)
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None


class MovieDetail(MovieSummary):
    """A single movie with its strongest tags and graph metrics."""

    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    top_tags: list[str] = Field(default_factory=list)
    pagerank: Optional[float] = None
    community: Optional[int] = None


class ScoredMovie(MovieSummary):
    """A movie with a ranking score and the reason the score exists."""

    score: float
    detail: Optional[str] = None


class Page(BaseModel):
    """Pagination envelope."""

    limit: int
    offset: int
    count: int


class ScoredMovieList(BaseModel):
    """A ranked list of movies plus how it was computed."""

    method: str
    page: Page
    items: list[ScoredMovie]


class UserProfile(BaseModel):
    """What the graph knows about one user's taste."""

    user_id: int
    rating_count: int
    avg_rating: Optional[float]
    favourite_genres: list[dict[str, Any]]
    community: Optional[int] = None


class EvidencePath(BaseModel):
    """One concrete chain of graph facts supporting a recommendation."""

    kind: Literal["similar_users", "similar_movie", "genre_affinity", "shared_tag"]
    weight: float
    nodes: list[str]
    description: str


class WhyRecommendation(BaseModel):
    """Answer to "why was this recommended?" built only from graph paths."""

    user_id: int
    movie: MovieSummary
    already_rated: bool
    evidence: list[EvidencePath]
    summary: str
    source: Literal["graph"] = "graph"


class PathStep(BaseModel):
    """One movie on a taste chain."""

    movie_id: int
    title: str
    similarity_to_next: Optional[float] = None


class MoviePath(BaseModel):
    """Shortest chain of similar movies between two movies."""

    method: str
    hops: int
    total_cost: Optional[float] = None
    steps: list[PathStep]


class PredictionResult(BaseModel):
    """ML estimate that a user will like a movie."""

    user_id: int
    movie_id: int
    probability: float
    model: str
    note: str = "Model estimate, not an observed rating."


class AskRequest(BaseModel):
    """A natural-language question for the NL-to-Cypher assistant."""

    question: str = Field(
        min_length=3, max_length=500, examples=["Which 5 sci-fi movies from the 90s are rated highest?"]
    )


class CypherCheckRequest(BaseModel):
    """A Cypher query to run through the guardrails (not executed)."""

    cypher: str = Field(min_length=1, max_length=4000, examples=["MATCH (m:Movie) DETACH DELETE m"])


class GuardrailReport(BaseModel):
    """What the safety checks did with the generated query."""

    passed: bool
    checks: list[dict[str, Any]]


class AskResponse(BaseModel):
    """Answer from the NL-to-Cypher assistant, separating facts from AI text."""

    question: str
    status: Literal["answered", "rejected", "error"]
    cypher: Optional[str] = None
    attempts: int
    guardrails: list[GuardrailReport]
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Verified data returned by Neo4j")
    answer: Optional[str] = Field(None, description="AI-written summary of `rows` (may contain mistakes)")
    answer_grounded: Optional[bool] = Field(None, description="True if every number in the answer appears in `rows`")
    trace: list[str] = Field(default_factory=list, description="Steps the agent took")


class HealthResponse(BaseModel):
    """Service health."""

    status: Literal["ok", "degraded"]
    neo4j: bool
    llm: bool


class AlgorithmRunResult(BaseModel):
    """Summary of a graph-algorithm recomputation."""

    results: dict[str, Any]
    seconds: float
