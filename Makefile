.PHONY: help venv install data seed seed-large gds train eval benchmark run docker-build docker-run docker-down \
        docker-push test test-unit lint format clean tree notebook

TAG ?= cinegraph-api:dev
PY := .venv/bin/python

help:
	@echo "CineGraph — commands:"
	@echo "  make venv           Create local virtualenv (.venv)"
	@echo "  make install        Install requirements (incl. dev tools) into .venv"
	@echo "  make data           Download MovieLens datasets into data/raw"
	@echo "  make docker-run     Start the Docker Compose stack (Neo4j, API, Nginx) in the background"
	@echo "  make seed           Load DATASET (default ml-latest-small) into Neo4j"
	@echo "  make seed-large     Load 2M ratings from ml-25m instead"
	@echo "  make gds            Run graph algorithms (similarity, PageRank, Louvain, FastRP)"
	@echo "  make train          Evaluate + train the link-prediction model (needs make gds)"
	@echo "  make eval           Evaluate the NL-to-Cypher assistant (needs Ollama)"
	@echo "  make benchmark      Before/after index & query benchmarks (PROFILE)"
	@echo "  make run            Run FastAPI locally with reload on :8001"
	@echo "  make test           Run all tests with coverage (integration tests need the stack)"
	@echo "  make test-unit      Run only tests that need no database"
	@echo "  make lint           Run pylint (must score >= 9.5)"
	@echo "  make format         Run black"
	@echo "  make docker-build   Build the API image (TAG=$(TAG))"
	@echo "  make docker-push    Push the API image (set TAG=youruser/cinegraph-api:1.0)"
	@echo "  make docker-down    Stop the stack"
	@echo "  make notebook       Open the demo notebook"
	@echo "  make clean          Remove caches and temp files"
	@echo "  make tree           Show project tree (depth 3)"

venv:
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
		. .venv/bin/activate && pip install --upgrade pip; \
		echo "Created .venv"; \
	else echo ".venv already exists"; fi
	@echo "To activate: source .venv/bin/activate"

install: venv
	@. .venv/bin/activate && pip install -r requirements-dev.txt

data:
	@mkdir -p data/raw
	@cd data/raw && for d in ml-latest-small ml-25m; do \
		if [ ! -d $$d ]; then curl -fsSLO https://files.grouplens.org/datasets/movielens/$$d.zip && unzip -q $$d.zip && rm $$d.zip; \
		else echo "$$d already present"; fi; done

seed:
	@$(PY) -m scripts.seed_data

seed-large:
	@$(PY) -m scripts.seed_data --dataset ml-25m --max-ratings 2000000 --reset

gds:
	@$(PY) -c "from app.database.neo4j_client import get_client; from app.services.gds_service import GdsService; \
	import json; r, s = GdsService(get_client()).run_all(); print(json.dumps(r, indent=2)); print(f'{s:.1f}s')"

train:
	@$(PY) -m scripts.train_link_model

eval:
	@OLLAMA_URL=$${OLLAMA_URL_HOST:-http://localhost:11434} $(PY) -m scripts.evaluate_agent

benchmark:
	@$(PY) -m scripts.benchmark

run:
	@. .venv/bin/activate && OLLAMA_URL=http://localhost:11434 uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

docker-build:
	docker compose build

docker-run:
	docker compose up -d --build
	@echo "API (via Nginx): http://localhost:8080/docs   Neo4j Browser: http://localhost:7474"

docker-down:
	docker compose down

docker-push:
	docker tag cinegraph-api:dev $(TAG)
	docker push $(TAG)

test:
	@. .venv/bin/activate && pytest -q --cov=app --cov-report=term-missing --cov-report=xml

test-unit:
	@. .venv/bin/activate && pytest -q -m "not integration and not containers" --cov=app --cov-report=term-missing

lint:
	@. .venv/bin/activate && pylint app scripts tests --fail-under=9.5

format:
	@. .venv/bin/activate && black app scripts tests -l 120

notebook:
	@. .venv/bin/activate && jupyter notebook demo.ipynb

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \; || true
	find . -type f -name "*.pyc" -delete || true
	rm -rf .pytest_cache .coverage coverage.xml htmlcov

tree:
	@if command -v tree >/dev/null 2>&1; then \
		tree -L 3 -I "node_modules|dist|.git|.venv|__pycache__|data"; \
	else find . -maxdepth 3 -type d -not -path '*/\.*' -not -path './data*' | sort; fi
