# Project: Agentic Document Extraction

## Objective
Design and implement a vision-first, agentic document extraction system as a FastAPI service in modern Python 3.12 using a clean, test-first workflow. The system enables users to submit documents via HTTP API along with a JSON schema and extract structured information. The system intelligently processes text-based documents (txt, csv, etc.) directly while treating visual documents (PDF, images, presentations) as visual objects where meaning is encoded in layout, structure, and spatial relationships. The system uses agentic patterns (plan, decide, act, verify) with LangChain orchestration to iteratively improve extraction quality until output meets defined thresholds.

- Complete the next most important task in `./TASKS.md` according to the objective. It does not have to be the next task in the order.
- Review the `./AGENT.md` file to understand how to work on this project
- ONLY complete one task. Do NOT attempt to complete multiple tasks.
- There is no need to worry about breaking changes and backwards compatibility for this project.
- The task is not completed until:
  - Tests for the task are comprehensive and passing
  - There are no obvious TODO/FIXME comments left in critical paths for that task
  - All acceptance criteria are met and the following commands succeed without errors:
  - `uv run ruff check .`
  - `uv run ruff format .` (no changes or only expected formatting changes)
  - `uv run mypy src`
  - `uv run pytest` 
  - `uv run pytest --cov=src --cov-report=term-missing`
  - Run the service locally and manually verify the feature works as expected end-to-end
- The task must be marked as complete (`[x]`) in `./TASKS.md`
- The completed task is to be committed to git with a descriptive commit message referencing the task

## Adding New Tasks
If you determine that a task is missing or needs to be broken down further, please do so in the TASKS.md file.

## Final Completion Criteria
When all of the tasks in TASKS.md are complete, the project is considered done so output `<promise>COMPLETE</promise>