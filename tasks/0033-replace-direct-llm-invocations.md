# Replace Direct LLM Invocations with LangChain Agents

- [ ] Replace Direct LLM Invocations with LangChain Agents
  - As a developer, I want to route all LLM/VLM calls through LangChain agent abstractions instead of direct model invocation, so that orchestration is consistent and extensible.
  - **Acceptance Criteria**:
    - Audit the codebase for direct LLM/VLM calls (e.g., raw OpenAI client usage or direct `ChatOpenAI.invoke`) and list replacements
    - Replace direct calls with LangChain agent abstractions (AgentExecutor or equivalent) using tools where appropriate
    - Prefer using the LangChain `create_agent` helper when constructing agent workflows
    - Ensure text extraction, reading order, visual extraction, and synthesis paths use the agent abstraction
    - Preserve model configuration and prompt behavior
    - Update tests to cover agent-based invocation paths
    - Logging and metadata capture agent run details for tracing
