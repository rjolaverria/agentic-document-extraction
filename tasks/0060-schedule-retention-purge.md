# Task 0060: Schedule Retention Purge Task

## Objective
Ensure the retention purge task runs automatically on the configured interval across API and worker deployments.

## Acceptance Criteria
- [ ] Docket/cron wiring invokes `retention_purge_task` at `ADE_RETENTION_CHECK_INTERVAL_HOURS`.
- [ ] Works for both local (memory) and Redis-backed deployments; no duplicate purge when multiple workers run.
- [ ] Configuration and run instructions documented in README/API docs.
- [ ] Tests or smoke check verifying scheduling hook is invoked (can be a small unit test or integration stub).

## Notes
- Prefer using Docket’s periodic scheduling if available; otherwise document a cronjob/worker entrypoint.
- Ensure purge respects `ADE_RETENTION_DAYS` and logs metrics.
