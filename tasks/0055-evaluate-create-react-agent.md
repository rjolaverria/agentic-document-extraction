# Task 0055: Evaluate create_react_agent Migration from langgraph.prebuilt

## Objective
Investigate whether migrating from `langchain.agents.create_agent` to `langgraph.prebuilt.create_react_agent` would benefit the extraction agent architecture.

## Context
The current code uses `create_agent` from `langchain.agents`:

```python
from langchain.agents import AgentState, create_agent
...
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=ExtractionAgentState if tools else None,
    response_format=response_schema,
)
```

LangGraph's `create_react_agent` is the modern, recommended approach:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(
    llm,
    tools,
    checkpointer=checkpointer  # Optional for memory
)
```

## Investigation Questions

### 1. Current Implementation Analysis
- [ ] What is `langchain.agents.create_agent`? Is it a LangGraph-based agent or older LangChain agent?
- [ ] How does it handle tool calling loops?
- [ ] How does it handle structured output (`response_format`)?
- [ ] What state management does it provide?

### 2. create_react_agent Capabilities
- [ ] Does `create_react_agent` support structured output (response_format)?
- [ ] How does it handle custom state schemas?
- [ ] What is the tool calling loop behavior?
- [ ] Does it support system prompts?
- [ ] What checkpointing/memory options are available?

### 3. Migration Considerations
- [ ] Would migration require significant code changes?
- [ ] Are there breaking changes in the API?
- [ ] How would structured JSON output be handled?
- [ ] Can we preserve the verification/refinement loop?

## Benefits of create_react_agent

1. **Durable Execution**: Agents persist through failures
2. **Human-in-the-Loop**: Inspect and modify state at any point
3. **Memory**: Short-term and long-term memory across sessions
4. **Checkpointing**: Save and resume agent state
5. **Better Observability**: Integrated with LangSmith

## Potential Code Changes

### Current Pattern
```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    state_schema=ExtractionAgentState if tools else None,
    response_format=response_schema,
)
result = agent.invoke(invoke_input)
```

### Potential New Pattern
```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_react_agent(
    llm.bind(response_format=response_schema),  # Structured output on model
    tools,
    checkpointer=checkpointer,
    state_schema=ExtractionAgentState,
)

config = {"configurable": {"thread_id": document_id}}
result = await agent.ainvoke({"messages": [...]}, config=config)
```

## Files to Investigate
- `src/agentic_document_extraction/agents/extraction_agent.py`
- LangGraph documentation for `create_react_agent`
- LangChain documentation for `create_agent`

## Acceptance Criteria
- [ ] Document findings on current `create_agent` implementation
- [ ] Document `create_react_agent` capabilities and limitations
- [ ] Provide recommendation: migrate or keep current implementation
- [ ] If migrating, create implementation plan with specific code changes
- [ ] Identify any blockers or risks

## Output
This task produces a **recommendation document**, not code changes. Based on findings:

1. **If migration recommended**: Create follow-up implementation task
2. **If keeping current**: Document reasons and close task

## Dependencies
None - this is an independent investigation task.

## Notes
- This is a lower priority task compared to the core refactoring (Tasks 0051-0054)
- The current implementation may already be using LangGraph under the hood
- Focus on whether migration provides tangible benefits for this use case
- Consider: Is the effort worth the potential benefits?
