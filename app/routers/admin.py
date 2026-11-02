"""Protected maintenance endpoints and health checks."""

from fastapi import APIRouter, Depends

from app.agent.llm import get_llm
from app.database.neo4j_client import Neo4jClient, get_client
from app.dependencies import gds_service, require_admin
from app.models.schemas import AlgorithmRunResult, HealthResponse
from app.services.gds_service import GdsService

router = APIRouter(tags=["admin"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(client: Neo4jClient = Depends(get_client)):
    """Neo4j and LLM reachability. The API is 'degraded' (still serving graph endpoints) if only the LLM is down."""
    neo4j_ok = client.verify()
    llm_ok = get_llm().is_available()
    return HealthResponse(status="ok" if neo4j_ok else "degraded", neo4j=neo4j_ok, llm=llm_ok)


@router.get("/api/admin/gds/status", dependencies=[Depends(require_admin)], summary="Graph-algorithm outputs")
def gds_status(service: GdsService = Depends(gds_service)):
    """How many nodes/relationships carry algorithm results."""
    return service.status()


@router.post(
    "/api/admin/gds/run",
    response_model=AlgorithmRunResult,
    dependencies=[Depends(require_admin)],
    summary="Recompute graph algorithms (requires X-API-Key)",
)
def gds_run(service: GdsService = Depends(gds_service)):
    """Re-run Node Similarity, PageRank, Louvain and FastRP. Expensive, so it is key-protected."""
    results, seconds = service.run_all()
    return AlgorithmRunResult(results=results, seconds=round(seconds, 2))
