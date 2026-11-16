.PHONY: sync lint typecheck test ui bench serve

sync:
	uv sync --python 3.13 --all-extras

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest --cov=aether_compiler --cov-report=term-missing --cov-fail-under=95

ui:
	cd ui && npm ci && npm run build

bench:
	uv run aether compile-ir
	uv run aether plan-arena
	uv run aether autotune-tiles
	uv run aether eval-ollama

serve:
	uv run aether serve --host 127.0.0.1 --port 8080
