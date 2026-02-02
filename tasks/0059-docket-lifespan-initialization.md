# Task 0059: Inline Docket Initialization in Lifespan

## Objective
Start and manage the Docket client and worker inside the FastAPI lifespan so the API bootstraps background task processing without a separate `docket worker` command.

## Context
Right now Docket must be launched independently. The FastMCP `LifespanMixin` shows a pattern where a Docket instance is created inside the application lifespan, registered via contextvars, and a worker is started with `run_forever()`. Adopting this pattern will simplify operations (single process/command) and ensure task registration happens alongside API startup.

## Acceptance Criteria
- [ ] FastAPI lifespan creates a Docket instance using existing settings and starts a Worker (async context) when the server starts; it shuts down cleanly on server stop.
- [ ] Docket/Worker references are stored in shared context (app state or contextvars) so dependencies and tasks can access them without global imports.
- [ ] Task registration happens during startup using the same registry used by the current standalone worker path; no separate CLI step is required.
- [ ] Concurrency, name, and connection settings reuse existing configuration (env/settings) with sane defaults; production/non-production behavior documented.
- [ ] Health/log output indicates when the worker is running or skipped (e.g., Docket not installed or tasks disabled).
- [ ] Integration test or smoke test proves that starting the API automatically processes a queued task without launching an external worker process.
- [ ] README / API docs updated to remove the standalone worker step and describe the new lifecycle.

## Proposed Implementation Notes
- Add a lifespan context manager (or FastAPI `lifespan` function) that mirrors FastMCP’s approach: create Docket, register tasks, set contextvars, start a Worker with `run_forever()`, cancel it on shutdown.
- Centralize contextvars or app state helpers (e.g., `current_docket`, `current_worker`) so background utilities and request handlers can retrieve the running instances.
- Keep a guard so if Docket is unavailable or tasks are disabled, the server still starts but skips worker startup.
- Ensure the shutdown path cancels the worker task, awaits cancellation, and releases contextvars to avoid leaked connections in tests.
