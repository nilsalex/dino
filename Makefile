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

run:
	uv run python -m src.main

run-playwright:
	USE_PLAYWRIGHT=true BROWSER_URL=http://localhost:5173 BROWSER_TYPE=firefox GAME=reaction uv run python -m src.train_local

run-local:
	uv run python -m src.train_local
