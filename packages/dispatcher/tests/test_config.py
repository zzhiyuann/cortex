"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import yaml

from dispatcher.config import Config


def test_defaults():
    """Config with no file uses sensible defaults."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"telegram:\n  bot_token: test123\n  chat_id: 42\n")
        f.flush()
        cfg = Config(f.name)

    assert cfg.bot_token == "test123"
    assert cfg.chat_id == 42
    assert cfg.agent_command == "claude"
    assert cfg.max_concurrent == 3
    assert cfg.timeout == 1800

    os.unlink(f.name)


def test_env_override():
    """Env vars override file values."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"telegram:\n  bot_token: file_token\n  chat_id: 1\n")
        f.flush()

        os.environ["DISPATCHER_BOT_TOKEN"] = "env_token"
        os.environ["DISPATCHER_CHAT_ID"] = "99"
        try:
            cfg = Config(f.name)
            assert cfg.bot_token == "env_token"
            assert cfg.chat_id == 99
        finally:
            del os.environ["DISPATCHER_BOT_TOKEN"]
            del os.environ["DISPATCHER_CHAT_ID"]
            os.unlink(f.name)


def test_project_routes():
    """Projects config generates keyword routes."""
    data = {
        "telegram": {"bot_token": "t", "chat_id": 1},
        "projects": {
            "myapp": {
                "path": "/tmp/myapp",
                "keywords": ["app", "frontend"],
            }
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        f.flush()

        cfg = Config(f.name)
        routes = cfg.get_project_routes()

        assert "myapp" in routes
        assert "app" in routes
        assert "frontend" in routes
        assert routes["app"] == Path("/tmp/myapp")

    os.unlink(f.name)


def test_validate_missing_token():
    """Validation catches missing required fields."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"{}\n")
        f.flush()
        cfg = Config(f.name)

    errors = cfg.validate()
    assert any("bot_token" in e for e in errors)
    assert any("chat_id" in e for e in errors)

    os.unlink(f.name)
