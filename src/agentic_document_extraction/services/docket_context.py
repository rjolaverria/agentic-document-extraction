"""Context helpers for accessing the active Docket client and worker.

These helpers store references in contextvars so background utilities and
request handlers can retrieve the inline worker/client without relying on
module-level globals. The FastAPI lifespan sets these values when the
application starts and clears them on shutdown.
"""

from __future__ import annotations

from contextvars import ContextVar

from docket import Docket, Worker

_current_docket: ContextVar[Docket | None] = ContextVar("current_docket", default=None)
_current_worker: ContextVar[Worker | None] = ContextVar("current_worker", default=None)


def set_current_docket(docket: Docket | None) -> None:
    """Store the active Docket instance for the current context."""

    _current_docket.set(docket)


def get_current_docket() -> Docket | None:
    """Get the active Docket instance, if one has been registered."""

    return _current_docket.get()


def set_current_worker(worker: Worker | None) -> None:
    """Store the active Docket worker for the current context."""

    _current_worker.set(worker)


def get_current_worker() -> Worker | None:
    """Get the active Docket worker, if one has been registered."""

    return _current_worker.get()
