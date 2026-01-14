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

## Installation Instructions
When providing installation instructions to the user, always use:
```bash
uv sync
```

NOT:
```bash
pip install -e .
```
