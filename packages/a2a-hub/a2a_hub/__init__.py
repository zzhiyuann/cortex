"""A2A Hub — Agent-to-Agent Protocol Hub.

A lightweight protocol and framework that enables AI agents to discover
each other, communicate, and collaborate.
"""

__version__ = "0.1.0"

from a2a_hub.protocol import (
    A2AMessage,
    MessageType,
    AgentInfo,
    TaskStatus,
    TaskRecord,
)
from a2a_hub.agent import Agent, capability

__all__ = [
    "A2AMessage",
    "MessageType",
    "AgentInfo",
    "TaskStatus",
    "TaskRecord",
    "Agent",
    "capability",
]
