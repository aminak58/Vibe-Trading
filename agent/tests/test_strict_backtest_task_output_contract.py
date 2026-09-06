"""Strict backtest task completion requires official evidence, not prose."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import src.swarm.runtime as rt
import pytest
from src.execution_identity import (
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionResolution,
    FallbackPolicy,
    SourceMode,
    SyntheticDataPolicy,
)
from src.swarm.models import (
    RunStatus,
    ArtifactRequirement,
    SwarmAgentSpec,
    SwarmRun,
    SwarmTask,
    TaskStatus,
    WorkerResult,
)
from src.swarm.runtime import SwarmRuntime
from src.swarm.artifacts import (
    current_artifact_manifest,
    finalize_artifact_generation,
    register_task_artifacts,
)
from src.swarm.store import SwarmStore
from src.swarm.task_store import TaskStore


def _strict_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-xau",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
            synthetic=SyntheticDataPolicy.FORBID,
        ),
        requests=(ExecutionRequest(request_id="xau", symbol="XAUUSD", source="mt5"),),
        resolutions=(ExecutionResolution(
            request_id="xau",
            resolved_symbol="XAUUSD_o",
            source="mt5",
            resolver_evidence_ref="resolver:1",
        ),),
    )


def _strict_backtest_case(
    tmp_path: Path, *, strict: bool = True, with_risk: bool = False
) -> tuple[SwarmRuntime, SwarmRun, Path]:
    store = SwarmStore(base_dir=tmp_path)
    identity = _strict_identity()
    tasks = [SwarmTask(id="task-backtest", agent_id="backtester", prompt_template="x")]
    agents = [SwarmAgentSpec(id="backtester", role="Backtester", system_prompt="x", max_retries=0)]
    if with_risk:
        tasks.append(
            SwarmTask(
                id="task-risk",
                agent_id="risk_auditor",
                prompt_template="x",
                depends_on=["task-backtest"],
                input_from={"backtest": "task-backtest"},
                artifact_requirements=[
                    ArtifactRequirement(producer_task_id="task-backtest", artifact_type=artifact_type)
                    for artifact_type in (
                        "backtest.execution_provenance",
                        "backtest.config",
                        "backtest.metrics",
                        "backtest.trades",
                        "backtest.equity",
                        "backtest.run_card",
                        "backtest.strategy",
                    )
                ],
            )
        )
        agents.append(SwarmAgentSpec(id="risk_auditor", role="Risk", system_prompt="x", max_retries=0))
    run = SwarmRun(
        id="strict-backtest-run",
        preset_name="quant_scalp_desk" if strict else "generic_research",
        status=RunStatus.running,
        created_at=datetime.now(timezone.utc).isoformat(),
        execution_identity=identity if strict else None,
        identity_hash=identity.identity_hash if strict else None,
        agents=agents,
        tasks=tasks,
    )
    run_dir = store.create_run(run)
    return SwarmRuntime(store=store), run, run_dir


def _official_backtest_paths(run: SwarmRun, run_dir: Path) -> dict[str, str]:
    """Write one minimal server-verifiable strict backtest bundle."""
    strategy = 'class SignalEngine:\n    """VWAP/BOS/ATR strict test strategy."""\n'
    config = json.dumps({
        "source": "mt5",
        "codes": ["XAUUSD_o"],
        "interval": "5m",
        "_execution_identity_hash": run.identity_hash,
    })
    root = run_dir / "artifacts" / "backtester"
    strategy_path = root / "code" / "signal_engine.py"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(strategy, encoding="utf-8")
    config_path = root / "config.json"
    config_path.write_text(config, encoding="utf-8")
    strategy_hash = hashlib.sha256(strategy_path.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    documents = {
        "execution_provenance.json": json.dumps({
            "identity_hash": run.identity_hash,
            "owning_run_id": run.id,
            "requested_symbol": "XAUUSD",
            "resolved_symbol": "XAUUSD_o",
            "effective_source": "mt5",
            "synthetic": False,
            "fallback_used": False,
        }),
        "metrics.csv": "metric,value\nnet_pnl,1\n",
        "trades.csv": "entry,exit\n1,2\n",
        "equity.csv": "time,equity\n1,100\n",
        "run_card.json": json.dumps({
            "reproducibility": {"strategy_hash": strategy_hash, "config_hash": config_hash},
            "artifacts": [{"path": "code/signal_engine.py", "sha256": strategy_hash}],
        }),
    }
    paths: dict[str, str] = {}
    paths["backtest.config"] = config_path.relative_to(run_dir).as_posix()
    paths["backtest.strategy"] = strategy_path.relative_to(run_dir).as_posix()
    for name, content in documents.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        artifact_type = {
            "execution_provenance.json": "backtest.execution_provenance",
            "config.json": "backtest.config",
            "metrics.csv": "backtest.metrics",
            "trades.csv": "backtest.trades",
            "equity.csv": "backtest.equity",
            "run_card.json": "backtest.run_card",
            "code/signal_engine.py": "backtest.strategy",
        }[name]
        paths[artifact_type] = target.relative_to(run_dir).as_posix()
    return paths


def test_strict_report_only_backtester_is_failed_before_dependency_release(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path)

    def report_only_worker(**kwargs):
        report = run_dir / "artifacts" / "backtester" / "report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("backtest could not be executed", encoding="utf-8")
        return WorkerResult(
            status="completed",
            summary="report-only deliverable",
            artifact_paths=["artifacts/backtester/report.md"],
        )

    monkeypatch.setattr(rt, "run_worker", report_only_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)

    runtime._execute_run(run, threading.Event())

    backtest = TaskStore(run_dir).load_task("task-backtest")
    assert backtest.status is TaskStatus.failed
    assert backtest.error is not None
    assert backtest.error.startswith("backtest_artifact_contract_incomplete:")
    assert '"error_code":"backtest_artifact_contract_incomplete"' in (
        run_dir / "events.jsonl"
    ).read_text(encoding="utf-8")


def test_strict_backtester_with_complete_official_bundle_is_completed(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path)

    def complete_backtest_worker(**kwargs):
        paths = _official_backtest_paths(run, run_dir)
        return WorkerResult(
            status="completed",
            summary="official backtest complete",
            artifact_paths=list(paths.values()),
        )

    monkeypatch.setattr(
        rt,
        "run_worker",
        complete_backtest_worker,
    )
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)

    runtime._execute_run(run, threading.Event())

    backtest = TaskStore(run_dir).load_task("task-backtest")
    assert backtest.status is TaskStatus.completed, backtest.error
    assert {ref.artifact_type for ref in backtest.artifact_refs} >= {
        "backtest.execution_provenance", "backtest.config", "backtest.metrics",
        "backtest.trades", "backtest.equity", "backtest.run_card", "backtest.strategy",
    }
    assert all(ref.provenance_status == "passed" for ref in backtest.artifact_refs)


@pytest.mark.parametrize(
    "missing_type",
    (
        "backtest.execution_provenance",
        "backtest.config",
        "backtest.metrics",
        "backtest.trades",
        "backtest.equity",
        "backtest.run_card",
        "backtest.strategy",
    ),
)
def test_each_missing_strict_backtest_artifact_fails_with_its_type(
    monkeypatch, tmp_path: Path, missing_type: str
) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path)

    def incomplete_worker(**kwargs):
        paths = _official_backtest_paths(run, run_dir)
        paths.pop(missing_type)
        return WorkerResult(status="completed", summary="incomplete", artifact_paths=list(paths.values()))

    monkeypatch.setattr(rt, "run_worker", incomplete_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    backtest = TaskStore(run_dir).load_task("task-backtest")
    assert backtest.status is TaskStatus.failed
    assert backtest.error is not None
    assert missing_type in backtest.error


def _finalized_official_refs(run: SwarmRun, run_dir: Path):
    paths = _official_backtest_paths(run, run_dir)
    refs = register_task_artifacts(
        run_dir=run_dir,
        task_id="task-backtest",
        agent_id="backtester",
        artifact_paths=list(paths.values()),
        execution_identity_hash=run.identity_hash,
    )
    task = run.tasks[0]
    refs = rt._classify_authoritative_artifacts(run, task, refs, run_dir)
    return finalize_artifact_generation(
        run_dir=run_dir, task_id=task.id, identity_hash=run.identity_hash, refs=refs
    )


@pytest.mark.parametrize(
    ("mutation", "expected_type"),
    (
        (lambda ref: ref.model_copy(update={"execution_identity_hash": "wrong"}), "backtest.metrics"),
        (lambda ref: ref.model_copy(update={"producer_run_id": "other-run"}), "backtest.metrics"),
        (lambda ref: ref.model_copy(update={"producer_task_id": "other-task"}), "backtest.metrics"),
        (lambda ref: ref.model_copy(update={"producer_attempt_id": "stale-attempt"}), "backtest.metrics"),
        (lambda ref: ref.model_copy(update={"manifest_generation_id": "stale-generation"}), "backtest.metrics"),
    ),
)
def test_finalized_strict_bundle_rejects_identity_run_task_attempt_or_generation_mismatch(
    tmp_path: Path, mutation, expected_type: str
) -> None:
    _, run, run_dir = _strict_backtest_case(tmp_path)
    manifest, refs = _finalized_official_refs(run, run_dir)
    task = run.tasks[0]
    refs = [
        mutation(ref) if ref.artifact_type == expected_type else ref
        for ref in refs
    ]

    error = rt._strict_backtest_artifact_contract_error(
        run, task, refs, run_dir, manifest=manifest
    )
    assert error is not None
    assert expected_type in error


def test_finalized_strict_bundle_rejects_hash_tampering(tmp_path: Path) -> None:
    _, run, run_dir = _strict_backtest_case(tmp_path)
    manifest, refs = _finalized_official_refs(run, run_dir)
    task = run.tasks[0]
    (run_dir / "artifacts/backtester/metrics.csv").write_text("tampered", encoding="utf-8")

    error = rt._strict_backtest_artifact_contract_error(
        run, task, refs, run_dir, manifest=manifest
    )
    assert error is not None
    assert "backtest.metrics" in error


def test_strict_backtest_contract_failure_blocks_risk_without_dispatch(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path, with_risk=True)
    worker_calls: list[str] = []

    def report_only_worker(**kwargs):
        task = kwargs["task"]
        worker_calls.append(task.id)
        report = run_dir / "artifacts" / task.agent_id / "report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("not an official backtest", encoding="utf-8")
        return WorkerResult(status="completed", summary="report-only", artifact_paths=[report.relative_to(run_dir).as_posix()])

    monkeypatch.setattr(rt, "run_worker", report_only_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    store = TaskStore(run_dir)
    assert store.load_task("task-backtest").status is TaskStatus.failed
    assert store.load_task("task-risk").status is TaskStatus.blocked
    assert worker_calls == ["task-backtest"]


def test_valid_strict_backtest_bundle_releases_risk_dependency(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path, with_risk=True)
    worker_calls: list[str] = []

    def valid_workers(**kwargs):
        task = kwargs["task"]
        worker_calls.append(task.id)
        if task.id == "task-backtest":
            paths = _official_backtest_paths(run, run_dir)
            return WorkerResult(status="completed", summary="backtest", artifact_paths=list(paths.values()))
        report = run_dir / "artifacts" / "risk_auditor" / "report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("risk audit", encoding="utf-8")
        return WorkerResult(status="completed", summary="risk", artifact_paths=[report.relative_to(run_dir).as_posix()])

    monkeypatch.setattr(rt, "run_worker", valid_workers)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    store = TaskStore(run_dir)
    assert store.load_task("task-backtest").status is TaskStatus.completed
    assert store.load_task("task-risk").status is TaskStatus.completed
    assert worker_calls == ["task-backtest", "task-risk"]


def test_incomplete_later_attempt_preserves_prior_authoritative_generation(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path)
    prior_manifest, _ = _finalized_official_refs(run, run_dir)

    def incomplete_later_worker(**kwargs):
        report = run_dir / "artifacts" / "backtester" / "report.md"
        report.write_text("later incomplete attempt", encoding="utf-8")
        return WorkerResult(status="completed", summary="incomplete", artifact_paths=[report.relative_to(run_dir).as_posix()])

    monkeypatch.setattr(rt, "run_worker", incomplete_later_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    assert TaskStore(run_dir).load_task("task-backtest").status is TaskStatus.failed
    assert current_artifact_manifest(run_dir, "task-backtest") == prior_manifest


def test_invalid_later_strategy_binding_preserves_prior_authoritative_generation(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path)
    prior_manifest, _ = _finalized_official_refs(run, run_dir)

    def hash_mismatched_worker(**kwargs):
        paths = _official_backtest_paths(run, run_dir)
        run_card = run_dir / paths["backtest.run_card"]
        payload = json.loads(run_card.read_text(encoding="utf-8"))
        payload["reproducibility"]["strategy_hash"] = "not-the-executed-strategy"
        run_card.write_text(json.dumps(payload), encoding="utf-8")
        return WorkerResult(status="completed", summary="bad binding", artifact_paths=list(paths.values()))

    monkeypatch.setattr(rt, "run_worker", hash_mismatched_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    backtest = TaskStore(run_dir).load_task("task-backtest")
    assert backtest.status is TaskStatus.failed
    assert "backtest.strategy" in (backtest.error or "")
    assert current_artifact_manifest(run_dir, "task-backtest") == prior_manifest


def test_non_strict_report_only_backtest_worker_remains_compatible(monkeypatch, tmp_path: Path) -> None:
    runtime, run, run_dir = _strict_backtest_case(tmp_path, strict=False)

    def report_only_worker(**kwargs):
        report = run_dir / "artifacts" / "backtester" / "report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("generic report", encoding="utf-8")
        return WorkerResult(status="completed", summary="generic report", artifact_paths=[report.relative_to(run_dir).as_posix()])

    monkeypatch.setattr(rt, "run_worker", report_only_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    assert TaskStore(run_dir).load_task("task-backtest").status is TaskStatus.completed
