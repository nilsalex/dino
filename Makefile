install:
	uv sync

lock:
	uv lock

lint:
	uv run mypy src/
	uv run ruff format --check src/
	uv run ruff check src/

format:
	uv run ruff format src/
	uv run ruff check --fix src/
