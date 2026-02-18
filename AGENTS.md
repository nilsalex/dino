# DINO RL - Agent Coding Instructions

## Project Overview

A Chrome Dino game reinforcement learning agent using DQN. Captures game frames via GStreamer, processes them through a CNN, and trains locally without Ray.

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

# Run a specific module
uv run python -m src.train_local
uv run python -m src.main

# Run a single Python file
uv run python path/to/file.py
```

**Note**: No test suite currently exists in this repository.

## Project Structure

```
src/
├── agents/          # RL agent implementations (LocalDQNTrainer)
├── buffers/         # Thread-safe experience buffers
├── capture/         # Frame capture (GStreamer) and preprocessing
├── core/            # Config, types, constants
├── env/             # Game interface, episode management
├── models/          # Neural network models
├── replay/          # Experience replay buffer
├── training/        # Training threads
└── utils/           # Logging, metrics, visualization
```

## Code Style Guidelines

### Imports

- Use `from __future__ import annotations` for forward references
- Use `TYPE_CHECKING` guard for forward type imports that would cause circular imports
- GStreamer requires: `gi.require_version("Gst", "1.0")` before imports
- Standard library imports first, then third-party, then local `from src.*`
- Avoid `from module import *`
- Use `# noqa: E402` for imports after side-effect code (e.g., after `load_dotenv()`)

### Type Annotations

- **Required** for all functions and methods
- Use modern syntax: `dict | None` instead of `Optional[Dict]`
- Use `list[T]`, `dict[K, V]` syntax, not `List[T]`, `Dict[K, V]`
- For numpy/cv2 arrays: use `NDArray[Any]` when exact dtype is unknown
- Include return types even for `None`
- Use `object` for generic args that accept any type in stubs/callbacks

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
- Check for `None` before operations: `if obj is None: raise RuntimeError(...)`
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
- Use `namedtuple` for immutable record-like types (see `Experience` in `core/types.py`)
- Base classes in `base.py` files define interfaces with `raise NotImplementedError`
- `__init__.py` files are typically empty
- Private methods prefixed with `_`
- Use properties with `@property` decorator for computed attributes
- Place type definitions in separate files (e.g., `core/types.py`)
- Configuration belongs in `core/config.py` and related files

### Docstrings

- Use triple quotes: `"""Method summary."""` for simple methods
- Use Args/Returns sections for complex methods
- Keep docstrings concise and informative

### Comments

- Minimize comments - code should be self-documenting
- Use comments only to explain **why**, not **what**

### Device Handling

- Prefer config-based: `config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Use type annotation: `torch.device | None`

### Environment Variables

- Use `os.getenv("KEY", "default")` for environment variables
- Define in `Config` dataclass with type conversion (see `.env.example`)

### GStreamer/GObject Integration

- Always require version before imports: `gi.require_version("Gst", "1.0")`
- Use `# type: ignore[arg-type]` or `[union-attr]` for Gi introspection limitations

## Architecture Notes

- **Config**: Single `Config` dataclass in `src/core/config.py` holds all hyperparameters
- **Types**: Type aliases in `src/core/types.py` (Frame, State, Experience, etc.)
- **Agent**: `LocalDQNTrainer` implements DQN with target network
- **Capture**: GStreamer pipeline captures frames, `FrameProcessor` handles stacking/preprocessing
- **Replay**: `ReplayBuffer` stores experiences for off-policy training
