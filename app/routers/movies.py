"""Movie endpoints: lookup, search, similarity, rankings, taste paths."""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.dependencies import movie_service
from app.models.schemas import MovieDetail, MoviePath, MovieSummary, Page, ScoredMovie, ScoredMovieList
from app.services.movie_service import MovieService

router = APIRouter(prefix="/api/movies", tags=["movies"])

Limit = Query(10, ge=1, le=50, description="Page size (max 50)")
Offset = Query(0, ge=0, le=1000, description="Rows to skip")


def to_list(method: str, rows: list[dict], limit: int, offset: int) -> ScoredMovieList:
    """Wrap service rows in the paginated response model."""
    items = [ScoredMovie(**row["movie"], score=row["score"], detail=row.get("detail")) for row in rows]
    return ScoredMovieList(method=method, page=Page(limit=limit, offset=offset, count=len(items)), items=items)


@router.get("/search", response_model=list[MovieSummary], summary="Search movies by title")
def search(
    q: str = Query(..., min_length=2, max_length=100, examples=["matrix"]),
    limit: int = Limit,
    offset: int = Offset,
    service: MovieService = Depends(movie_service),
):
    """Full-text title search, tolerant to small typos for single words."""
    return [MovieSummary(**row["movie"]) for row in service.search_movies(q, limit, offset)]


@router.get("/trending", response_model=ScoredMovieList, summary="Ranked movies (recent, central or top-rated)")
def trending(
    metric: Literal["recent", "pagerank", "rating"] = "recent",
    window_days: int = Query(365, ge=7, le=3650, description="For metric=recent"),
    min_ratings: int = Query(20, ge=1, le=1000),
    limit: int = Limit,
    offset: int = Offset,
    service: MovieService = Depends(movie_service),
):
    """`recent`: most-loved in the last window of the dataset. `pagerank`: taste-network centrality (GDS).
    `rating`: best average with at least `min_ratings` ratings."""
    rows = service.trending(metric, window_days, min_ratings, limit, offset)
    return to_list(metric, rows, limit, offset)


@router.get("/path", response_model=MoviePath, summary="Chain of similar movies linking two movies")
def taste_path(
    from_id: int = Query(..., alias="from", ge=1, examples=[1]),
    to_id: int = Query(..., alias="to", ge=1, examples=[858]),
    weighted: bool = Query(False, description="False: fewest hops (shortestPath). True: strongest links (Dijkstra)"),
    service: MovieService = Depends(movie_service),
):
    """Path-finding over SIMILAR relationships."""
    return service.taste_path(from_id, to_id, weighted)


@router.get("/{movie_id}", response_model=MovieDetail, summary="Movie details")
def get_movie(movie_id: int, service: MovieService = Depends(movie_service)):
    """One movie with tags and graph metrics."""
    return service.get_movie(movie_id)


@router.get("/{movie_id}/similar", response_model=ScoredMovieList, summary="Similar movies")
def similar(
    movie_id: int,
    method: Literal["collaborative", "content"] = "collaborative",
    limit: int = Limit,
    offset: int = Offset,
    service: MovieService = Depends(movie_service),
):
    """`collaborative`: liked by the same people (GDS Node Similarity). `content`: shared genres and tags."""
    return to_list(method, service.similar_movies(movie_id, method, limit, offset), limit, offset)
