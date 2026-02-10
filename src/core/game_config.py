"""Game-specific configurations."""

from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration for a specific game."""

    n_actions: int
    frame_stack: int
    frame_skip: int
    action_keys: list[int]
    action_names: list[str]
    action_keys_str: list[str] | None = None


GAME_CONFIGS: dict[str, GameConfig] = {
    "dino": GameConfig(
        n_actions=2,
        frame_stack=4,
        frame_skip=4,
        action_keys=[0, 103],
        action_names=["NO_ACTION", "JUMP_UP"],
        action_keys_str=["", " "],
    ),
    "reaction": GameConfig(
        n_actions=3,
        frame_stack=1,
        frame_skip=1,
        action_keys=[0, 30, 48],
        action_names=["DO_NOTHING", "PRESS_A", "PRESS_B"],
        action_keys_str=["", "a", "b"],
    ),
}


def get_game_config(game_name: str = "dino") -> GameConfig:
    """Get configuration for a specific game.

    Args:
        game_name: Name of the game ("dino" or "reaction").

    Returns:
        GameConfig object with game-specific settings.

    Raises:
        ValueError: If game_name is not recognized.
    """
    if game_name not in GAME_CONFIGS:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(GAME_CONFIGS.keys())}")
    return GAME_CONFIGS[game_name]
