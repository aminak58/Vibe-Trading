"""Security and DAG-edge coverage for dependency-scoped artifact reads."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.swarm.artifacts import ReadDependencyArtifactTool
from src.swarm.models import ArtifactRef, RunStatus, SwarmAgentSpec, SwarmRun, SwarmTask, TaskStatus, WorkerResult
from src.swarm.runtime import SwarmRuntime
from src.swarm.store import SwarmStore
from src.swarm.task_store import TaskStore


def _ref(run_dir: Path, task_id: str, agent_id: str, filename: str, content: str, *, identity_hash: str | None = "identity") -> ArtifactRef:
    path = run_dir / "artifacts" / agent_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    relative = path.relative_to(run_dir).as_posix()
    return ArtifactRef(
        artifact_id=f"artifact-{task_id}-{filename}",
        producer_run_id=run_dir.name,
        producer_task_id=task_id,
        producer_agent_id=agent_id,
        run_relative_path=relative,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        byte_size=path.stat().st_size,
        execution_identity_hash=identity_hash,
    )


def _payload(result: str) -> dict:
    return json.loads(result)


def test_reader_allows_only_registered_dependency_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    allowed = _ref(run_dir, "backtest", "backtester", "metrics.csv", "metric,value\ntrades,17\n")
    sibling = _ref(run_dir, "risk", "risk_auditor", "report.md", "not a dependency")
    tool = ReadDependencyArtifactTool(
        run_dir=run_dir,
        allowed_refs={allowed.artifact_id: allowed},
        execution_identity_hash="identity",
    )

    ok = _payload(tool.execute(artifact_id=allowed.artifact_id, limit=1))
    denied = _payload(tool.execute(artifact_id=sibling.artifact_id))

    assert ok["status"] == "ok"
    assert ok["content"] == "metric,value\n"
    assert denied["error_code"] == "dependency_artifact_denied"


def test_reader_rejects_cross_run_traversal_tamper_and_identity_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    ref = _ref(run_dir, "backtest", "backtester", "metrics.csv", "metric,value\ntrades,17\n")
    tool = ReadDependencyArtifactTool(
        run_dir=run_dir,
        allowed_refs={ref.artifact_id: ref},
        execution_identity_hash="identity",
    )

    # A server bug or stale record pointing at another run is rejected even if
    # the opaque ID was otherwise edge-authorized.
    cross_run = ref.model_copy(update={"producer_run_id": "run-other"})
    assert _payload(
        ReadDependencyArtifactTool(
            run_dir=run_dir, allowed_refs={cross_run.artifact_id: cross_run}, execution_identity_hash="identity"
        ).execute(artifact_id=cross_run.artifact_id)
    )["error_code"] == "dependency_artifact_denied"

    traversal = ref.model_copy(update={"run_relative_path": "artifacts/backtester/../../other.txt"})
    assert _payload(
        ReadDependencyArtifactTool(
            run_dir=run_dir, allowed_refs={traversal.artifact_id: traversal}, execution_identity_hash="identity"
        ).execute(artifact_id=traversal.artifact_id)
    )["error_code"] == "dependency_artifact_invalid"

    (run_dir / ref.run_relative_path).write_text("changed", encoding="utf-8")
    assert _payload(tool.execute(artifact_id=ref.artifact_id))["error_code"] == "provenance_conflict"

    mismatch = ref.model_copy(update={"execution_identity_hash": "other"})
    assert _payload(
        ReadDependencyArtifactTool(
            run_dir=run_dir, allowed_refs={mismatch.artifact_id: mismatch}, execution_identity_hash="identity"
        ).execute(artifact_id=mismatch.artifact_id)
    )["error_code"] == "provenance_conflict"


def test_runtime_propagates_refs_only_for_declared_input_from_edge(tmp_path: Path) -> None:
    store = SwarmStore(base_dir=tmp_path / "swarm")
    runtime = SwarmRuntime(store=store)
    producer = SwarmAgentSpec(id="producer", role="p", system_prompt="x", tools=[], skills=[])
    other = SwarmAgentSpec(id="other", role="o", system_prompt="x", tools=[], skills=[])
    consumer = SwarmAgentSpec(id="consumer", role="c", system_prompt="x", tools=[], skills=[])
    task_a = SwarmTask(id="a", agent_id="producer", prompt_template="a")
    task_b = SwarmTask(id="b", agent_id="other", prompt_template="b")
    task_c = SwarmTask(
        id="c", agent_id="consumer", prompt_template="c", depends_on=["a", "b"],
        input_from={"declared": "a"},
    )
    run = SwarmRun(
        id="run-edge", preset_name="test", status=RunStatus.pending,
        created_at=datetime.now(timezone.utc).isoformat(), agents=[producer, other, consumer],
        tasks=[task_a, task_b, task_c], identity_hash="identity",
    )
    store.create_run(run)
    run_dir = store.run_dir(run.id)
    store_a = TaskStore(run_dir)
    refs_a = [_ref(run_dir, "a", "producer", "metrics.csv", "metric,value\n")]
    refs_b = [_ref(run_dir, "b", "other", "secret.txt", "not exposed")]
    store_a.save_task(task_a.model_copy(update={"status": TaskStatus.completed, "summary": "A", "artifact_refs": refs_a}))
    store_a.save_task(task_b.model_copy(update={"status": TaskStatus.completed, "summary": "B", "artifact_refs": refs_b}))
    store_a.save_task(task_c)
    captured: dict = {}

    def fake_worker(**kwargs):
        captured.update(kwargs)
        return WorkerResult(status="completed", summary="C")

    with patch.object(runtime, "_run_worker_with_retries", side_effect=fake_worker):
        runtime._execute_layer(
            run=run,
            task_store=store_a,
            agent_map={item.id: item for item in run.agents},
            layer_task_ids=["c"],
            task_summaries={"a": "A", "b": "B"},
            run_dir=run_dir,
            cancel_event=threading.Event(),
        )

    assert set(captured["upstream_artifacts"]) == {"declared"}
    assert captured["upstream_artifacts"]["declared"] == refs_a
    assert captured["upstream_summaries"] == {"declared": "A"}
