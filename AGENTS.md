# DINO RL - Agent Coding Instructions

## Package Management

**ALWAYS use `uv` for Python package management** - this is a strict requirement. Never use pip, poetry, or pipenv.

## Essential Commands

```bash
# Install dependencies
make install

# Format code (ruff format + ruff check --fix)
make format

# Lint code (mypy + ruff format check + ruff check)
make lint

# Lock dependencies
make lock

# Run main application
make run

# Run headless training (Xvfb + Xlib)
make run-headless

# Run local training
make run-local
```

**Note**: No test suite currently exists in this repository.

## Code Style Guidelines

### Imports

- Use `from __future__ import annotations` for forward references
- Use `TYPE_CHECKING` guard for forward type imports that would cause circular imports
- GStreamer requires: `gi.require_version("Gst", "1.0")` before imports
- Standard library imports first, then third-party, then local `from src.*`
- Avoid `from module import *`

### Type Annotations

- **Required** for all functions and methods
- Use modern syntax: `dict | None` instead of `Optional[Dict]`
- Use `list[T]`, `dict[K, V]` syntax, not `List[T]`, `Dict[K, V]`
- For numpy/cv2 arrays: use `NDArray[Any]` when exact dtype is unknown
- Include return types even for `None`
- Use `object` for generic args that accept any type in stubs

### Formatting (Ruff)

- Line length: 120 characters
- Target Python version: 3.14
- Ruff selects: E, W, F, I, B, C4, UP, SIM, RUF
- **Before committing**: Run `make format` and `make lint`

### Naming Conventions

- Classes: `PascalCase` (e.g., `LocalDQNTrainer`)
- Functions/methods: `snake_case` (e.g., `get_state`, `update_target`)
- Variables: `snake_case` (e.g., `frame_processor`, `epsilon`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_EPISODES`)
- Private members: `_leading_underscore` (e.g., `_build_model`, `_preprocess`)
- Type aliases: `PascalCase` (e.g., `Frame`, `Experience`)

### Error Handling

- Prefer specific exceptions over generic `Exception`
- Use `try/except/finally` for resource cleanup
- Use `raise ValueError(...)` for invalid arguments with descriptive message
- Use `raise RuntimeError(...)` for invalid state
- Check for `None` before operations with: `if obj is None: raise RuntimeError(...)`
- Use try/except with specific exceptions: `except (ValueError, KeyError) as e:`

### Type Checking (Mypy)

- **Fix underlying issues, don't silence warnings**
- Use `# type: ignore[specific-error]` only for:
  - Incomplete third-party type stubs (GStreamer, PyGObject)
  - Dynamic attributes (e.g., Gi introspection)
- Always add explanatory comment after type ignores
- Never add rules to `disable_error_code` in pyproject.toml without approval
- Current exception: `disable_error_code = ["import-untyped"]`

### Code Organization

- Use dataclasses for configuration and simple data containers
- Use `namedtuple` for immutable record-like types
- Base classes in `base.py` files define interfaces with `raise NotImplementedError`
- `__init__.py` files are typically empty
- Private methods prefixed with `_`
- Use properties with `@property` decorator for computed attributes
- Place type definitions in separate files (e.g., `core/types.py`)
- Configuration belongs in `core/config.py` and related files

### Docstrings

- Use triple quotes for docstrings
- Format: `"""Method summary."""` for simple methods
- Format with Args/Returns sections for complex methods:
  ```python
  def method(arg: type) -> Return:
      """Method summary.

      Args:
          arg: Description.

      Returns:
          Description.
      """
  ```
- Keep docstrings concise and informative

### Comments

- Minimize comments - code should be self-documenting
- Use comments only to explain **why**, not **what**
- No code block separators like `# ---` or `# =====`

### Device Handling

- Prefer config-based device setting: `config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Import torch with `import torch`
- Use type annotation: `torch.device | None`

### Environment Variables

- Use `os.getenv("KEY", "default")` for environment variables
- Define in `Config` dataclass with type conversion:
  ```python
  headless: bool = os.getenv("HEADLESS", "").lower() == "true"
  ```

### GStreamer/GObject Integration

- Always require version before imports: `gi.require_version("Gst", "1.0")`
- Use `# type: ignore[arg-type]` or `[union-attr]` for Gi introspection limitations
- PyGObject stubs include some gaps requiring type ignores