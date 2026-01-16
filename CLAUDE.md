# Claude Code Instructions

## Package Manager
- **ALWAYS** use `uv` for Python package management
- **NEVER** use `pip`, `poetry`, `pipenv`, or any other package managers
- This is a strict requirement for this project

## Common Commands

This project uses a Makefile for common tasks. **ALWAYS** use these make commands:

- **Install dependencies**: `make install`
- **Format code**: `make format`
- **Lint code**: `make lint`
- **Lock dependencies**: `make lock`

### Direct UV Commands (if needed)
- Add a package: `uv add <package>`
- Add a dev dependency: `uv add --dev <package>`
- Run Python scripts: `uv run python <script>`
- Run commands in environment: `uv run <command>`

## Linting and Formatting

**Before committing code changes**, run:
```bash
make format
make lint
```

This will:
1. Format code with ruff
2. Fix linting issues automatically with ruff
3. Run mypy for type checking

## Installation Instructions
When providing installation instructions to the user, always use:
```bash
make install
```

NOT:
```bash
pip install -e .
uv sync
```
