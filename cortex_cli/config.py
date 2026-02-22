"""Shared configuration for the Cortex ecosystem.

Central config at ~/.cortex/config.yaml that all projects can read from.
Eliminates duplication of Telegram credentials, project paths, and service
endpoints across projects.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CORTEX_DIR = Path.home() / ".cortex"
CONFIG_PATH = CORTEX_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "version": 1,
    "telegram": {
        "bot_token": "",
        "chat_id": 0,
    },
    "projects": {
        # name: path — auto-populated by cortex init
    },
    "services": {
        "a2a_hub": {
            "host": "localhost",
            "port": 8765,
        },
        "dispatcher": {
            "max_concurrent": 3,
            "timeout": 1800,
        },
    },
    "paths": {
        "sessions": str(Path.home() / ".vibe-replay" / "sessions"),
        "forge_tools": str(Path.home() / ".forge" / "tools"),
        "logs": str(CORTEX_DIR),
        "errors": str(CORTEX_DIR / "errors.log"),
    },
}


def load_config() -> dict:
    """Load the shared Cortex config from ~/.cortex/config.yaml.

    Returns defaults merged with whatever is on disk.
    """
    config = _deep_copy(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            disk = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            _deep_merge(config, disk)
        except (yaml.YAMLError, OSError):
            pass
    return config


def save_config(config: dict):
    """Write config to ~/.cortex/config.yaml."""
    CORTEX_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def get_telegram_creds() -> tuple[str, int]:
    """Get Telegram bot_token and chat_id from shared config.

    Returns ("", 0) if not configured.
    """
    config = load_config()
    tg = config.get("telegram", {})
    return tg.get("bot_token", ""), tg.get("chat_id", 0)


def get_project_paths() -> dict[str, Path]:
    """Get a mapping of project name → project directory."""
    config = load_config()
    return {
        name: Path(path)
        for name, path in config.get("projects", {}).items()
        if path
    }


def get_service_config(service: str) -> dict:
    """Get config for a specific service (a2a_hub, dispatcher)."""
    config = load_config()
    return config.get("services", {}).get(service, {})


def _deep_merge(base: dict, override: dict):
    """Merge override into base (in-place). Nested dicts are merged recursively."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _deep_copy(d: dict) -> dict:
    """Simple deep copy for nested dicts with primitive values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy(v)
        else:
            result[k] = v
    return result
