# Investigation: create_react_agent Migration (Task 0055)

## Summary
Reviewed LangChain/LangGraph guidance on agent construction to determine whether migrating from `langchain.agents.create_agent` to `langgraph.prebuilt.create_react_agent` provides benefits for the extraction agent.

## Findings
- LangGraph v1 documentation explicitly deprecates `create_react_agent` in favor of LangChain's `create_agent`, and states `create_agent` runs on LangGraph. This means the current implementation already sits on the modern LangGraph runtime.
- LangChain v1 migration guidance recommends `langchain.agents.create_agent` instead of `langgraph.prebuilt.create_react_agent`, with API differences handled via `system_prompt` and middleware.
- LangChain agent docs and structured output docs confirm `create_agent` supports `response_format` for structured output and returns data in `structured_response`, which matches our current usage. Custom state via `state_schema` is supported for `AgentState`.
- Examples of `create_agent` with a checkpointer show that state persistence and durability can be added without switching to `create_react_agent`, so LangGraph benefits are already available when needed.

## Recommendation
Do not migrate. The project already uses the recommended `langchain.agents.create_agent` API that is built on LangGraph, supports `response_format`, and integrates with custom state via `state_schema`. Migrating to the deprecated `langgraph.prebuilt.create_react_agent` would add an extra dependency without functional gain and risks losing first-class structured output handling.

## Implementation Plan
None. No follow-up task required unless we decide to add a checkpointer or middleware-based customization in the future.
