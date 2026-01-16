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

### Critical: Importance of Lints and Type Checks

**NEVER disable linting or type checking rules without explicit user approval.**

This project maintains strict code quality standards:

1. **Fix the underlying issue, don't silence the warning**
   - When mypy reports a type error, fix the types - don't add `# type: ignore` everywhere
   - When ruff reports unused variables, prefix them with `_` - don't disable the check
   - When a linting rule is violated, refactor the code to comply

2. **Only use ignores as a last resort**
   - `# type: ignore[specific-error]` is acceptable ONLY for:
     - Incomplete third-party type stubs (e.g., PyGObject, GStreamer)
     - Dynamic attributes that are intentionally set (e.g., `root.photo` in tkinter)
   - Always add a comment explaining WHY the ignore is necessary
   - Always use specific error codes, never bare `# type: ignore`

3. **Do not globally disable checks**
   - Never add rules to `disable_error_code` in pyproject.toml without user approval
   - Never add files to exclude lists to bypass linting
   - Each `# type: ignore` should be justified and localized

4. **Type annotations are required**
   - All functions and methods must have type annotations
   - Use proper types (not just `Any` everywhere)
   - Use `NDArray[Any]` for cv2/numpy when the exact dtype is hard to specify
   - Use modern syntax: `dict | None` instead of `Optional[Dict]`

5. **Address all lint failures before committing**
   - `make lint` must pass completely
   - Fix issues found by mypy, ruff format, and ruff check
   - If uncertain how to fix an issue, ask the user for guidance

**Remember**: These checks exist to catch bugs early and maintain code quality. Disabling them defeats their purpose.

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
