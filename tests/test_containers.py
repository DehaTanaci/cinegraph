"""Container health checks: the Docker Compose stack is up and its services answer.

Skipped when the stack is not running (e.g. in CI unit-test jobs).
"""

import json
import shutil
import subprocess
import urllib.request

import pytest

pytestmark = pytest.mark.containers

NEO4J_HTTP = "http://localhost:7474"
API_VIA_NGINX = "http://localhost:8080"


def _get(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec - local URLs only
        return response.status, response.read().decode()


def _container_health(name: str) -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    result = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", name],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"container {name} is not running (start with `make docker-run`)")
    return result.stdout.strip()


@pytest.mark.parametrize("name", ["cinegraph-neo4j", "cinegraph-api", "cinegraph-nginx"])
def test_container_healthy(name):
    assert _container_health(name) == "healthy"


def test_neo4j_http_endpoint():
    _container_health("cinegraph-neo4j")
    status, body = _get(NEO4J_HTTP)
    assert status == 200 and "neo4j_version" in body


def test_api_health_through_reverse_proxy():
    _container_health("cinegraph-nginx")
    status, body = _get(f"{API_VIA_NGINX}/health")
    assert status == 200 and json.loads(body)["neo4j"] is True


def test_swagger_through_reverse_proxy():
    _container_health("cinegraph-nginx")
    status, body = _get(f"{API_VIA_NGINX}/docs")
    assert status == 200 and "swagger" in body.lower()
