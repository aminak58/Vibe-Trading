"""Adversarial coverage for generation-scoped strict artifact lineage."""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.swarm.runtime as rt
from src.swarm.models import (
    ArtifactManifest, ArtifactRef, ArtifactRequirement, RunStatus,
    SwarmAgentSpec, SwarmRun, SwarmTask, TaskStatus, WorkerResult,
)
from src.swarm.store import SwarmStore
from src.swarm.task_store import TaskStore


def _write_ref(rd: Path, *, artifact_id: str, task: str, agent: str, kind: str,
               generation: str, path: str, text: str = "x", run_id: str = "r",
               identity: str = "ih", derived: list[str] | None = None,
               derived_generation: str | None = None) -> ArtifactRef:
    target = rd / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return ArtifactRef(
        artifact_id=artifact_id, producer_run_id=run_id, producer_task_id=task,
        producer_agent_id=agent, run_relative_path=path,
        sha256=hashlib.sha256(text.encode()).hexdigest(), byte_size=len(text),
        execution_identity_hash=identity, artifact_type=kind,
        provenance_status="passed", producer_attempt_id="attempt-" + generation,
        manifest_generation_id=generation,
        derived_from_artifact_ids=derived or [],
        derived_from_manifest_generation=derived_generation,
    )


def _manifest(rd: Path, generation: str, artifact_ids: list[str]) -> None:
    root = rd / "artifact_manifests"
    root.mkdir(parents=True, exist_ok=True)
    manifest = ArtifactManifest(
        generation_id=generation, producer_attempt_id="attempt-" + generation,
        producer_task_id="task-backtest", run_id="r", identity_hash="ih",
        artifact_ids=artifact_ids,
    )
    (root / "task-backtest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )


def _case(tmp_path: Path, backtest_refs: list[ArtifactRef], risk_ref: ArtifactRef):
    store = SwarmStore(base_dir=tmp_path)
    runtime = rt.SwarmRuntime(store=store)
    agents = [
        SwarmAgentSpec(id="backtester", role="b", system_prompt="x", max_retries=0),
        SwarmAgentSpec(id="risk", role="r", system_prompt="x", max_retries=0),
    ]
    backtest = SwarmTask(
        id="task-backtest", agent_id="backtester", prompt_template="x",
        status=TaskStatus.completed, artifact_refs=backtest_refs,
    )
    risk = SwarmTask(
        id="task-risk", agent_id="risk", prompt_template="x",
        status=TaskStatus.completed, artifact_refs=[risk_ref],
    )
    report = SwarmTask(
        id="task-report", agent_id="risk", prompt_template="x",
        depends_on=["task-backtest", "task-risk"], blocked_by=[],
        input_from={"backtest": "task-backtest", "risk": "task-risk"},
        artifact_requirements=[
            ArtifactRequirement(producer_task_id="task-backtest", artifact_type="backtest.metrics"),
            ArtifactRequirement(producer_task_id="task-risk", artifact_type="risk.audit_report"),
        ],
    )
    run = SwarmRun(
        id="r", preset_name="quant_scalp_desk", status=RunStatus.running,
        created_at=datetime.now(timezone.utc).isoformat(), agents=agents,
        tasks=[backtest, risk, report], identity_hash="ih",
    )
    rd = store.run_dir(run.id)
    if rd.exists():
        for sub in ("tasks", "inboxes", "artifacts"):
            (rd / sub).mkdir(exist_ok=True)
        (rd / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    else:
        rd = store.create_run(run)
    ts = TaskStore(rd / "tasks")
    for task in (backtest, risk, report):
        ts.save_task(task)
    return store, runtime, run, ts


def _dispatch(runtime, store, run, ts, monkeypatch):
    calls = []
    monkeypatch.setattr(
        rt, "run_worker",
        lambda *a, **k: calls.append(a) or WorkerResult(status="completed", summary="ok"),
    )
    runtime._execute_layer(
        run=run, task_store=ts,
        agent_map={a.id: a for a in run.agents}, layer_task_ids=["task-report"],
        task_summaries={}, run_dir=store.run_dir("r"),
        cancel_event=threading.Event(), include_shell_tools=False, grounding_block="",
    )
    return calls, ts.load_task("task-report")


def _base_refs(rd: Path, *, generation="G2"):
    metric = _write_ref(
        rd, artifact_id="metric-" + generation, task="task-backtest", agent="backtester",
        kind="backtest.metrics", generation=generation,
        path="artifacts/backtester/metrics.csv", text="same-bytes",
    )
    risk = _write_ref(
        rd, artifact_id="risk-" + generation, task="task-risk", agent="risk",
        kind="risk.audit_report", generation="R2",
        path="artifacts/risk/report.md", text="risk",
        derived=[metric.artifact_id], derived_generation=generation,
    )
    return metric, risk


def test_stale_g1_lineage_rejected_after_g2_even_with_same_bytes(tmp_path, monkeypatch):
    rd = tmp_path / "r"
    rd.mkdir()
    g1, _ = _base_refs(rd, generation="G1")
    g2, _ = _base_refs(rd, generation="G2")
    # G1 and G2 intentionally point at the same path with identical bytes.
    # Hash validation alone therefore cannot distinguish the stale generation.
    stale_risk = _write_ref(
        rd, artifact_id="risk-stale", task="task-risk", agent="risk",
        kind="risk.audit_report", generation="R2",
        path="artifacts/risk/report.md", text="risk",
        derived=[g1.artifact_id], derived_generation="G1",
    )
    _manifest(rd, "G2", [g2.artifact_id])
    store, runtime, run, ts = _case(tmp_path, [g1, g2], stale_risk)
    calls, report = _dispatch(runtime, store, run, ts, monkeypatch)
    assert calls == []
    assert report.status is TaskStatus.blocked


def test_mixed_or_fabricated_generation_lineage_rejected(tmp_path, monkeypatch):
    rd = tmp_path / "r"; rd.mkdir()
    g1, _ = _base_refs(rd, generation="G1")
    g2, _ = _base_refs(rd, generation="G2")
    _manifest(rd, "G2", [g2.artifact_id])
    for derived, generation in [([g1.artifact_id, g2.artifact_id], "G2"), (["fabricated"], "G2"), ([g2.artifact_id], "FAKE")]:
        risk = _write_ref(
            rd, artifact_id="risk-" + generation + str(len(derived)), task="task-risk", agent="risk",
            kind="risk.audit_report", generation="R2", path="artifacts/risk/report.md",
            text="risk", derived=derived, derived_generation=generation,
        )
        store, runtime, run, ts = _case(tmp_path, [g1, g2], risk)
        calls, report = _dispatch(runtime, store, run, ts, monkeypatch)
        assert calls == []
        assert report.status is TaskStatus.blocked


def test_underlying_or_risk_report_tamper_blocks_dispatch(tmp_path, monkeypatch):
    rd = tmp_path / "r"; rd.mkdir()
    metric, risk = _base_refs(rd)
    _manifest(rd, "G2", [metric.artifact_id])
    store, runtime, run, ts = _case(tmp_path, [metric], risk)
    (store.run_dir("r") / metric.run_relative_path).write_text("tampered", encoding="utf-8")
    calls, report = _dispatch(runtime, store, run, ts, monkeypatch)
    assert calls == [] and report.status is TaskStatus.blocked

    # Restore a clean case, then tamper the risk report itself.
    tmp2 = tmp_path / "second"; rd2 = tmp2 / "r"; rd2.mkdir(parents=True)
    metric2, risk2 = _base_refs(rd2); _manifest(rd2, "G2", [metric2.artifact_id])
    store2, runtime2, run2, ts2 = _case(tmp2, [metric2], risk2)
    (store2.run_dir("r") / risk2.run_relative_path).write_text("tampered-risk", encoding="utf-8")
    calls2, report2 = _dispatch(runtime2, store2, run2, ts2, monkeypatch)
    assert calls2 == [] and report2.status is TaskStatus.blocked


def test_current_generation_exact_lineage_dispatches_report(tmp_path, monkeypatch):
    rd = tmp_path / "r"; rd.mkdir()
    metric, risk = _base_refs(rd)
    _manifest(rd, "G2", [metric.artifact_id])
    store, runtime, run, ts = _case(tmp_path, [metric], risk)
    calls, report = _dispatch(runtime, store, run, ts, monkeypatch)
    assert len(calls) == 1


def test_previous_run_or_wrong_producer_risk_report_rejected(tmp_path, monkeypatch):
    rd = tmp_path / "r"; rd.mkdir()
    metric, risk = _base_refs(rd)
    _manifest(rd, "G2", [metric.artifact_id])
    for update in (
        {"producer_run_id": "previous-run"},
        {"producer_task_id": "task-other"},
    ):
        bad = risk.model_copy(update=update)
        store, runtime, run, ts = _case(tmp_path, [metric], bad)
        calls, report = _dispatch(runtime, store, run, ts, monkeypatch)
        assert calls == []
        assert report.status is TaskStatus.blocked


def test_empty_lineage_rejected(tmp_path, monkeypatch):
    rd = tmp_path / "r"; rd.mkdir()
    metric, risk = _base_refs(rd)
    _manifest(rd, "G2", [metric.artifact_id])
    risk = risk.model_copy(update={"derived_from_artifact_ids": []})
    store, runtime, run, ts = _case(tmp_path, [metric], risk)
    calls, report = _dispatch(runtime, store, run, ts, monkeypatch)
    assert calls == []
    assert report.status is TaskStatus.blocked


def test_unfinalized_failed_attempt_does_not_replace_current_manifest(tmp_path):
    from src.swarm.artifacts import current_artifact_manifest

    rd = tmp_path / "r"; rd.mkdir()
    g1, _ = _base_refs(rd, generation="G1")
    _manifest(rd, "G1", [g1.artifact_id])
    before = current_artifact_manifest(rd, "task-backtest")
    assert before is not None and before.generation_id == "G1"

    # A later attempt may write candidate bytes, but until successful
    # finalization publishes a new manifest it has no authoritative generation.
    _write_ref(
        rd, artifact_id="candidate-g2", task="task-backtest", agent="backtester",
        kind="backtest.metrics", generation="G2",
        path="artifacts/backtester/candidate-g2.csv", text="failed-attempt",
    )
    after = current_artifact_manifest(rd, "task-backtest")
    assert after is not None and after.generation_id == "G1"
    assert after.artifact_ids == [g1.artifact_id]
