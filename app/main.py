"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.agent.llm import LLMError
from app.database.neo4j_client import close_client
from app.routers import admin, ask, movies, users
from app.services.movie_service import NotFoundError, PrerequisiteError

logger = logging.getLogger("cinegraph")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Close the Neo4j driver on shutdown."""
    yield
    close_client()


app = FastAPI(
    title="CineGraph API",
    version="1.0.0",
    description=(
        "Movie recommendations over a Neo4j graph of MovieLens ratings: collaborative filtering, "
        "graph-algorithm similarity and communities, explainable recommendations, an ML link-prediction "
        "model, and a guarded natural-language assistant."
    ),
    lifespan=lifespan,
)

app.include_router(movies.router)
app.include_router(users.router)
app.include_router(ask.router)
app.include_router(admin.router)

DEMO_PAGE = Path(__file__).parent / "static" / "demo.html"


@app.get("/", include_in_schema=False)
def root():
    """Send visitors to the demo page."""
    return RedirectResponse("/demo")


@app.get("/demo", include_in_schema=False)
def demo():
    """Single-page demo interface that calls the API from the browser."""
    return FileResponse(DEMO_PAGE)


@app.exception_handler(NotFoundError)
async def not_found(_: Request, exc: NotFoundError):
    """404 for unknown movie/user ids."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(PrerequisiteError)
async def prerequisite(_: Request, exc: PrerequisiteError):
    """409 when graph algorithms / the model have not been computed yet."""
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ServiceUnavailable)
async def db_down(_: Request, exc: ServiceUnavailable):
    """503 when Neo4j cannot be reached."""
    logger.error(f"Neo4j unavailable: {exc}")
    return JSONResponse(status_code=503, content={"detail": "Database unavailable, try again shortly"})


@app.exception_handler(Neo4jError)
async def db_error(_: Request, exc: Neo4jError):
    """500 with the Neo4j error code (no stack trace leaked)."""
    logger.error(f"Neo4j error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Database query failed", "code": exc.code})


@app.exception_handler(LLMError)
async def llm_error(_: Request, exc: LLMError):
    """502 when the language model fails."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})
