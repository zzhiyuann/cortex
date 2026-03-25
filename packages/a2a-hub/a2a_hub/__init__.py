"""A2A Hub — Agent-to-Agent Protocol Hub.

A lightweight protocol and framework that enables AI agents to discover
each other, communicate, and collaborate.
"""

__version__ = "0.1.0"

from a2a_hub.agent import Agent, capability
from a2a_hub.protocol import (
    A2AMessage,
    AgentInfo,
    MessageType,
    TaskRecord,
    TaskStatus,
)

__all__ = [
    "A2AMessage",
    "MessageType",
    "AgentInfo",
    "TaskStatus",
    "TaskRecord",
    "Agent",
    "capability",
]
