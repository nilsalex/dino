# Claude Code Instructions

## Package Manager
- **ALWAYS** use `uv` for Python package management
- **NEVER** use `pip`, `poetry`, `pipenv`, or any other package managers
- This is a strict requirement for this project

## Common Commands
- Install dependencies: `uv sync`
- Install with all extras: `uv sync --all-extras`
- Add a package: `uv add <package>`
- Add a dev dependency: `uv add --dev <package>`
- Run Python scripts: `uv run python <script>`
- Run commands in environment: `uv run <command>`

## Linting and Formatting
This project uses `ruff`, `mypy`, and `pylint` for code quality:

- **Format code**: `uv run ruff format <file>`
- **Lint code**: `uv run ruff check <file>`
- **Type check**: `uv run mypy <file>`
- **Full lint**: `uv run pylint <file>`

**Before committing code changes**, run:
```bash
uv run ruff format .
uv run ruff check --fix .
```

## Installation Instructions
When providing installation instructions to the user, always use:
```bash
uv sync
```

NOT:
```bash
pip install -e .
```
