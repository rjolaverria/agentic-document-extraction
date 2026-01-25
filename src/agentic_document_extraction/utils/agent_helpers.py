"""Helpers for LangChain agent execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)


def build_agent(
    model: Any,
    *,
    name: str | None = None,
    system_prompt: str | None = None,
    tools: Sequence[Any] | None = None,
) -> Any:
    """Create a LangChain agent graph with optional system prompt."""
    return create_agent(
        model=model,
        tools=list(tools or []),
        system_prompt=system_prompt,
        name=name,
    )


def invoke_agent(
    agent: Any,
    messages: list[BaseMessage],
    *,
    metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Invoke the agent and return the last AI message."""
    if metadata is not None:
        logger.debug("Invoking agent", extra={"metadata": metadata})
        result = agent.invoke({"messages": messages}, {"metadata": metadata})
    else:
        result = agent.invoke({"messages": messages})

    if not isinstance(result, dict):
        raise ValueError("Agent response must be a mapping with messages")

    raw_messages = result.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Agent response missing messages list")

    ai_message = _get_last_ai_message(raw_messages)
    if ai_message is None:
        raise ValueError("Agent response missing AI message")

    return ai_message


def _get_last_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def get_message_content(message: BaseMessage) -> str:
    """Normalize message content to a string."""
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def get_usage_metadata(message: BaseMessage) -> dict[str, Any]:
    """Extract usage metadata from a message."""
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return usage_metadata
    return {}
