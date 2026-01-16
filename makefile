install:
	uv sync

lock:
	uv lock

lint:
	uv run pylint src/
	uv run mypy src/
	uv run ruff format --check --diff src/
	uv run ruff check --diff src/

format:
	uv run ruff format src/
	uv run ruff check --fix src/
