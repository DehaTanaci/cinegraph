"""User endpoints: profile, recommendations, explanations, discovery, ML prediction."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.dependencies import movie_service, prediction_service
from app.models.schemas import EvidencePath, PredictionResult, ScoredMovieList, UserProfile, WhyRecommendation
from app.routers.movies import Limit, Offset, to_list
from app.services.movie_service import MovieService
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/{user_id}", response_model=UserProfile, summary="User taste profile")
def get_user(user_id: int, service: MovieService = Depends(movie_service)):
    """Rating stats and favourite genres."""
    return service.get_user(user_id)


@router.get("/{user_id}/recommendations", response_model=ScoredMovieList, summary="Personalised recommendations")
def recommendations(
    user_id: int,
    genre: Optional[str] = Query(None, examples=["Sci-Fi"]),
    limit: int = Limit,
    offset: int = Offset,
    service: MovieService = Depends(movie_service),
):
    """Collaborative filtering: movies loved by the users whose taste overlaps most with yours."""
    rows = service.recommendations(user_id, limit, offset, genre)
    return to_list("collaborative_filtering", rows, limit, offset)


@router.get(
    "/{user_id}/recommendations/{movie_id}/why",
    response_model=WhyRecommendation,
    summary="Why was this recommended? (graph evidence only)",
)
def why(user_id: int, movie_id: int, service: MovieService = Depends(movie_service)):
    """Explain a recommendation with actual graph paths: similar users, similar movies, genres, tags.

    No LLM is involved: every sentence comes from a path that exists in the database.
    """
    data = service.why_recommended(user_id, movie_id)
    evidence: list[EvidencePath] = []
    for p in data["peers"]:
        evidence.append(EvidencePath(
            kind="similar_users", weight=p["overlap"],
            nodes=[f"User {user_id}", *p["examples"], f"User {p['peer']}", data["movie"]["title"]],
            description=f"User {p['peer']} shares {p['overlap']} liked movies with you "
                        f"(e.g. {', '.join(p['examples'])}) and rated it {p['peerRating']}",
        ))
    for s in data["similar"]:
        evidence.append(EvidencePath(
            kind="similar_movie", weight=s["score"], nodes=[f"User {user_id}", s["liked"], data["movie"]["title"]],
            description=f"You rated {s['liked']} {s['rating']}; it has similarity {s['score']} with this movie",
        ))
    for g in data["genres"]:
        evidence.append(EvidencePath(
            kind="genre_affinity", weight=g["likedInGenre"], nodes=[f"User {user_id}", g["genre"],
                                                                     data["movie"]["title"]],
            description=f"You liked {g['likedInGenre']} {g['genre']} movies",
        ))
    for t in data["tags"]:
        evidence.append(EvidencePath(
            kind="shared_tag", weight=t["n"], nodes=[f"User {user_id}", *t["via"], t["tag"], data["movie"]["title"]],
            description=f"Tagged '{t['tag']}', like {', '.join(t['via'])} which you liked",
        ))
    if data["rated"] is not None:
        summary = f"You already rated this movie {data['rated']}."
    elif not evidence:
        summary = "No graph path connects you to this movie; it would not be recommended to you."
    else:
        parts = []
        if data["peers"]:
            parts.append(f"{len(data['peers'])} users with overlapping taste loved it")
        if data["similar"]:
            parts.append(f"it is similar to {data['similar'][0]['liked']}, which you liked")
        if data["genres"]:
            parts.append(f"you often like {data['genres'][0]['genre']} movies")
        summary = "Recommended because " + "; ".join(parts) + "." if parts else "Weak evidence: only shared tags."
    return WhyRecommendation(
        user_id=user_id, movie=data["movie"], already_rated=data["rated"] is not None,
        evidence=evidence, summary=summary,
    )


@router.get("/{user_id}/discover", response_model=ScoredMovieList, summary="Discover outside your usual taste")
def discover(user_id: int, limit: int = Limit, offset: int = Offset, service: MovieService = Depends(movie_service)):
    """Movies from Louvain taste communities you rarely watch, popular among people in your own community."""
    return to_list("louvain_community", service.discover(user_id, limit, offset), limit, offset)


@router.get("/{user_id}/predict/{movie_id}", response_model=PredictionResult, summary="ML: will this user like it?")
def predict(user_id: int, movie_id: int, service: PredictionService = Depends(prediction_service)):
    """Probability from the link-prediction model trained on graph embeddings (see scripts/train_link_model.py)."""
    return service.predict(user_id, movie_id)
