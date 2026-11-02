"""FastAPI dependencies: shared services and the admin API-key check."""

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.agent.graph import CypherAgent
from app.agent.llm import get_llm
from app.config import Settings, get_settings
from app.database.neo4j_client import Neo4jClient, get_client
from app.services.gds_service import GdsService
from app.services.movie_service import MovieService
from app.services.prediction_service import PredictionService


def movie_service(client: Neo4jClient = Depends(get_client)) -> MovieService:
    """Movie/user query service."""
    return MovieService(client)


def gds_service(client: Neo4jClient = Depends(get_client)) -> GdsService:
    """Graph-algorithm service."""
    return GdsService(client)


def prediction_service(client: Neo4jClient = Depends(get_client)) -> PredictionService:
    """ML link-prediction service."""
    return PredictionService(client)


def cypher_agent(client: Neo4jClient = Depends(get_client), settings: Settings = Depends(get_settings)) -> CypherAgent:
    """NL-to-Cypher workflow."""
    return CypherAgent(client, get_llm(), max_rows=settings.llm_max_rows)


def require_admin(
    x_api_key: str | None = Header(default=None, description="Admin API key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject the request unless X-API-Key matches ADMIN_API_KEY (constant-time comparison)."""
    if not settings.admin_api_key:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Admin endpoints are disabled: ADMIN_API_KEY unset")
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.admin_api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key header")
