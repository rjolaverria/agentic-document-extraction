## 2026-02-02 Inline Docket Initialization in FastAPI Lifespan

- Added contextvar helpers for current Docket/Worker references.
- Updated FastAPI lifespan to build Docket, register tasks, and start an inline Worker with configurable concurrency; cleans up on shutdown.
- Documented new single-process run flow in README and added instrumentation note.
- Added integration test proving inline worker processes queued task without external worker.
