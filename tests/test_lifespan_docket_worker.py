"""Integration test to ensure inline Docket worker starts with API lifespan."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from docket import Worker

from agentic_document_extraction.api import create_app
from agentic_document_extraction.docket_tasks import tasks as registered_tasks


@asynccontextmanager
async def create_test_client() -> AsyncIterator[httpx.AsyncClient]:
    """Create an async client with inline worker enabled in lifespan."""

    patch_targets = {
        "agentic_document_extraction.api.settings.docket_url": "memory://",
        "agentic_document_extraction.api.settings.docket_name": "lifespan-test",
        "agentic_document_extraction.api.settings.temp_upload_dir": "/tmp/ade_test",
    }

    with patch.dict("os.environ", {}, clear=False):
        for target, value in patch_targets.items():
            patch(target, value).start()
        try:
            app = create_app()
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                ) as client,
            ):
                client.app = app  # type: ignore[attr-defined]
                yield client
        finally:
            patch.stopall()


@pytest.mark.asyncio
async def test_inline_worker_processes_job(tmp_path: Path) -> None:
    """Verify a queued task is processed by the inline worker without external worker."""

    async def quick_task(flag: Path) -> str:
        flag.write_text("done")
        return "ok"

    async with create_test_client() as client:
        docket = client.app.state.docket  # type: ignore[attr-defined]

        # Register a quick test task and schedule it
        docket.register(quick_task)
        await docket.add(quick_task, key="inline-test-task")(tmp_path / "flag.txt")

        # Wait briefly for inline worker to pick it up
        async def wait_for_completion() -> None:
            for _ in range(40):
                execution = await docket.get_execution("inline-test-task")
                if execution and execution.state.value == "completed":
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("Task did not complete in time")

        await wait_for_completion()

        execution = await docket.get_execution("inline-test-task")
        assert execution is not None
        assert execution.state.value == "completed"
        assert (tmp_path / "flag.txt").read_text() == "done"

        # Ensure the worker reference is exposed
        worker: Worker = client.app.state.worker  # type: ignore[attr-defined]
        assert worker is not None
        assert any(task is quick_task for task in docket.tasks.values())
        assert any(task in registered_tasks for task in docket.tasks.values())
