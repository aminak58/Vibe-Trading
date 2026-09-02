"""Focused coverage for server-authoritative Swarm artifact registration."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.swarm.artifacts import register_task_artifacts
from src.swarm.models import RunStatus, SwarmAgentSpec, SwarmRun, SwarmTask, WorkerResult
from src.swarm.runtime import SwarmRuntime
from src.swarm.store import SwarmStore


def _write_worker_artifact(run_dir: Path, agent_id: str, name: str, text: str) -> Path:
    path = run_dir / "artifacts" / agent_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_registration_content_addresses_only_producer_owned_regular_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics = _write_worker_artifact(run_dir, "backtester", "metrics.csv", "metric,value\ntrades,4\n")
    _write_worker_artifact(run_dir, "risk", "report.md", "other worker")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = run_dir / "artifacts" / "backtester" / "outside-link.txt"
    link.symlink_to(outside)
    internal_link = run_dir / "artifacts" / "backtester" / "metrics-link.csv"
    internal_link.symlink_to(metrics)

    refs = register_task_artifacts(
        run_dir=run_dir,
        task_id="task-backtest",
        agent_id="backtester",
        artifact_paths=[
            "artifacts/backtester/metrics.csv",
            "artifacts/risk/report.md",
            "artifacts/backtester/outside-link.txt",
            "artifacts/backtester/metrics-link.csv",
            "../outside.txt",
            "artifacts/backtester/metrics.csv",
        ],
        execution_identity_hash="identity-123",
    )

    assert len(refs) == 1
    ref = refs[0]
    assert ref.producer_run_id == "run"
    assert ref.producer_task_id == "task-backtest"
    assert ref.producer_agent_id == "backtester"
    assert ref.run_relative_path == "artifacts/backtester/metrics.csv"
    assert ref.sha256 == hashlib.sha256(metrics.read_bytes()).hexdigest()
    assert ref.byte_size == metrics.stat().st_size
    assert ref.execution_identity_hash == "identity-123"
    assert ref.artifact_id.startswith("artifact-")


def test_task_legacy_artifacts_round_trip_without_typed_refs() -> None:
    task = SwarmTask.model_validate(
        {
            "id": "legacy",
            "agent_id": "analyst",
            "prompt_template": "x",
            "artifacts": ["artifacts/analyst/report.md"],
        }
    )
    assert task.artifact_refs == []
    assert SwarmTask.model_validate_json(task.model_dump_json()).artifacts == task.artifacts


def test_runtime_persists_refs_only_for_completed_task(tmp_path: Path) -> None:
    store = SwarmStore(base_dir=tmp_path / "swarm")
    runtime = SwarmRuntime(store=store)
    agent = SwarmAgentSpec(
        id="analyst", role="Analyst", system_prompt="x", tools=[], skills=[],
        max_iterations=1, timeout_seconds=5, max_retries=0,
    )
    task = SwarmTask(id="task-1", agent_id="analyst", prompt_template="x")
    run = SwarmRun(
        id="run-artifacts",
        preset_name="test",
        status=RunStatus.pending,
        created_at=datetime.now(timezone.utc).isoformat(),
        agents=[agent],
        tasks=[task],
        identity_hash="strict-hash",
    )
    store.create_run(run)
    run_dir = store.run_dir(run.id)
    artifact = _write_worker_artifact(run_dir, "analyst", "report.md", "real result")

    result = WorkerResult(
            status="completed",
            summary="real result",
            artifact_paths=["artifacts/analyst/report.md"],
            iterations=1,
    )

    with (
        patch.object(runtime, "_prefetch_grounding_data"),
        patch.object(runtime, "_execute_layer", return_value={"task-1": result}),
    ):
        runtime._execute_run(run, threading.Event())

    persisted = store.load_run(run.id)
    assert persisted is not None
    task_after = persisted.tasks[0]
    assert task_after.status.value == "completed"
    assert task_after.artifacts == ["artifacts/analyst/report.md"]
    assert len(task_after.artifact_refs) == 1
    assert task_after.artifact_refs[0].sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert task_after.artifact_refs[0].execution_identity_hash == "strict-hash"
