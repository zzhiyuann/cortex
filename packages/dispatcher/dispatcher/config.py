"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "dispatcher"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "telegram": {
        "bot_token": "",
        "chat_id": 0,
    },
    "agent": {
        "command": "claude",
        "args": ["-p", "--dangerously-skip-permissions"],
        "max_concurrent": 3,
        "timeout": 1800,
        "max_turns": 50,
        "max_turns_chat": 50,
        "max_turns_followup": 50,
        "question_timeout": 600,  # F7: auto-timeout unanswered agent questions (10 min)
    },
    "behavior": {
        "poll_timeout": 30,
        "progress_interval": 180,
        "recent_window": 300,
        "cancel_keywords": ["cancel", "stop"],
        "status_keywords": ["status"],
    },
    "projects": {},
    "data_dir": str(DEFAULT_CONFIG_DIR / "data"),
}


class Config:
    """Merged configuration from YAML + env vars."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else DEFAULT_CONFIG_FILE
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        merged = _deep_copy(DEFAULTS)

        # Load YAML if it exists
        if self._path.exists():
            with open(self._path) as f:
                file_data = yaml.safe_load(f) or {}
            _deep_merge(merged, file_data)

        # Env var overrides
        env_map = {
            "DISPATCHER_BOT_TOKEN": ("telegram", "bot_token"),
            "DISPATCHER_CHAT_ID": ("telegram", "chat_id"),
            "DISPATCHER_AGENT_COMMAND": ("agent", "command"),
            "DISPATCHER_MAX_CONCURRENT": ("agent", "max_concurrent"),
            "DISPATCHER_TIMEOUT": ("agent", "timeout"),
            "DISPATCHER_DATA_DIR": ("data_dir",),
        }
        for env_key, path in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                _set_nested(merged, path, _coerce(val))

        self._data = merged

    # -- Accessors --

    @property
    def bot_token(self) -> str:
        return self._data["telegram"]["bot_token"]

    @property
    def chat_id(self) -> int:
        return int(self._data["telegram"]["chat_id"])

    @property
    def agent_command(self) -> str:
        return self._data["agent"]["command"]

    @property
    def agent_args(self) -> list[str]:
        return list(self._data["agent"]["args"])

    @property
    def max_concurrent(self) -> int:
        return int(self._data["agent"]["max_concurrent"])

    @property
    def timeout(self) -> int:
        return int(self._data["agent"]["timeout"])

    @property
    def max_turns(self) -> int:
        return int(self._data["agent"]["max_turns"])

    @property
    def max_turns_chat(self) -> int:
        return int(self._data["agent"]["max_turns_chat"])

    @property
    def max_turns_followup(self) -> int:
        return int(self._data["agent"]["max_turns_followup"])

    @property
    def question_timeout(self) -> int:
        return int(self._data["agent"]["question_timeout"])

    @property
    def poll_timeout(self) -> int:
        return int(self._data["behavior"]["poll_timeout"])

    @property
    def progress_interval(self) -> int:
        return int(self._data["behavior"]["progress_interval"])

    @property
    def recent_window(self) -> int:
        return int(self._data["behavior"]["recent_window"])

    @property
    def cancel_keywords(self) -> set[str]:
        return set(self._data["behavior"]["cancel_keywords"])

    @property
    def status_keywords(self) -> set[str]:
        return set(self._data["behavior"]["status_keywords"])

    @property
    def projects(self) -> dict[str, dict]:
        return self._data["projects"]

    @property
    def data_dir(self) -> Path:
        return Path(self._data["data_dir"]).expanduser()

    @property
    def memory_file(self) -> Path:
        return self.data_dir / "memory.md"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if config is valid."""
        errors = []
        if not self.bot_token:
            errors.append("telegram.bot_token is required")
        if not self.chat_id:
            errors.append("telegram.chat_id is required")
        if not self.agent_command:
            errors.append("agent.command is required")
        return errors

    def get_project_routes(self) -> dict[str, Path]:
        """Build keyword -> path mapping from projects config."""
        routes: dict[str, Path] = {}
        for name, proj in self.projects.items():
            path = Path(proj["path"]).expanduser()
            routes[name] = path
            for kw in proj.get("keywords", []):
                routes[kw] = path
        return routes

    def raw(self) -> dict[str, Any]:
        return _deep_copy(self._data)


def _deep_copy(d: dict) -> dict:
    """Deep copy a config dict."""
    return copy.deepcopy(d)


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _set_nested(d: dict, keys: tuple, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _coerce(val: str):
    """Try to coerce string env var to int/bool."""
    if val.isdigit():
        return int(val)
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    return val
