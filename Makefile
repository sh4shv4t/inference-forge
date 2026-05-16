.PHONY: install dev test test-all lint fmt benchmark docker-up docker-down logs

install:
	poetry install

dev:
	poetry run uvicorn inference_forge.main:app --reload --port 8000 --app-dir src

test:
	poetry run pytest tests/unit -v

test-all:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check . && poetry run ruff format --check .

fmt:
	poetry run ruff format .

benchmark:
	poetry run python benchmarks/run_benchmark.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

logs:
	docker compose logs -f inference-forge
