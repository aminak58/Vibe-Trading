"""Strict risk-audit artifact emission is server-owned, not worker prose."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import src.swarm.runtime as rt
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
    ArtifactManifest,
    ArtifactRef,
    ArtifactRequirement,
    RunStatus,
    SwarmAgentSpec,
    SwarmRun,
    SwarmTask,
    TaskStatus,
    WorkerResult,
)
from src.swarm.runtime import SwarmRuntime
from src.swarm.store import SwarmStore
from src.swarm.task_store import TaskStore


def _strict_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-gold",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
            synthetic=SyntheticDataPolicy.FORBID,
        ),
        requests=(ExecutionRequest(request_id="gold", symbol="XAUUSD", source="mt5"),),
        resolutions=(
            ExecutionResolution(
                request_id="gold",
                resolved_symbol="XAUUSD_o",
                source="mt5",
                resolver_evidence_ref="resolver:1",
            ),
        ),
    )


def _backtest_ref(
    run_dir: Path, *, artifact_id: str, artifact_type: str, path: str
) -> ArtifactRef:
    target = run_dir / path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{artifact_id}\n".encode()
    target.write_bytes(payload)
    return ArtifactRef(
        artifact_id=artifact_id,
        producer_run_id="strict-risk-run",
        producer_task_id="task-backtest",
        producer_agent_id="backtester",
        run_relative_path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        execution_identity_hash="identity-1",
        artifact_type=artifact_type,
        provenance_status="passed",
        producer_attempt_id="attempt-backtest",
        manifest_generation_id="generation-backtest",
    )


def _runtime_case(
    tmp_path: Path, *, strict: bool = True, duplicate_metrics: bool = True
) -> tuple[SwarmRuntime, SwarmRun, TaskStore, Path]:
    store = SwarmStore(base_dir=tmp_path)
    agents = [
        SwarmAgentSpec(id="backtester", role="Backtester", system_prompt="x", max_retries=0),
        SwarmAgentSpec(id="risk_auditor", role="Risk", system_prompt="x", max_retries=0),
        SwarmAgentSpec(id="report_aggregator", role="Report", system_prompt="x", max_retries=0),
    ]
    backtest = SwarmTask(id="task-backtest", agent_id="backtester", prompt_template="x")
    risk = SwarmTask(
        id="task-risk",
        agent_id="risk_auditor",
        prompt_template="x",
        depends_on=["task-backtest"],
        input_from={"backtest": "task-backtest"},
        artifact_requirements=[
            ArtifactRequirement(producer_task_id="task-backtest", artifact_type=kind)
            for kind in (
                "backtest.execution_provenance",
                "backtest.metrics",
                "backtest.trades",
                "backtest.equity",
                "backtest.config",
            )
        ],
    )
    run = SwarmRun(
        id="strict-risk-run",
        preset_name="quant_scalp_desk",
        status=RunStatus.running,
        created_at=datetime.now(timezone.utc).isoformat(),
        agents=agents,
        tasks=[backtest, risk],
        execution_identity=_strict_identity() if strict else None,
        identity_hash=None if strict else "identity-1",
    )
    identity_hash = run.identity_hash
    run_dir = store.create_run(run)
    refs = [
        _backtest_ref(run_dir, artifact_id="provenance", artifact_type="backtest.execution_provenance", path="artifacts/backtester/execution_provenance.json"),
        _backtest_ref(run_dir, artifact_id="metrics-primary", artifact_type="backtest.metrics", path="artifacts/backtester/metrics.csv"),
        # A second valid metrics file reproduces the controlled-run defect.
        *(
            [
                _backtest_ref(
                    run_dir,
                    artifact_id="metrics-secondary",
                    artifact_type="backtest.metrics",
                    path="artifacts/backtester/run_5m/metrics.csv",
                )
            ]
            if duplicate_metrics
            else []
        ),
        _backtest_ref(run_dir, artifact_id="trades", artifact_type="backtest.trades", path="artifacts/backtester/trades.csv"),
        _backtest_ref(run_dir, artifact_id="equity", artifact_type="backtest.equity", path="artifacts/backtester/equity.csv"),
        _backtest_ref(run_dir, artifact_id="config", artifact_type="backtest.config", path="artifacts/backtester/config.json"),
    ]
    refs = [ref.model_copy(update={"execution_identity_hash": identity_hash}) for ref in refs]
    manifest_dir = run_dir / "artifact_manifests"
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / "task-backtest.json").write_text(
        ArtifactManifest(
            generation_id="generation-backtest",
            producer_attempt_id="attempt-backtest",
            producer_task_id="task-backtest",
            run_id=run.id,
            identity_hash=run.identity_hash,
            artifact_ids=[ref.artifact_id for ref in refs],
        ).model_dump_json(),
        encoding="utf-8",
    )
    task_store = TaskStore(run_dir / "tasks")
    backtest = backtest.model_copy(update={"status": TaskStatus.completed, "artifact_refs": refs})
    run.tasks = [backtest, risk]
    task_store.save_task(backtest)
    task_store.save_task(risk)
    return SwarmRuntime(store=store), run, task_store, run_dir


def test_strict_risk_worker_emits_typed_audit_report_with_duplicate_backtest_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    runtime, run, _, run_dir = _runtime_case(tmp_path)

    def completed_risk_worker(**kwargs):
        report = run_dir / "artifacts" / "risk_auditor" / "report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("official risk audit", encoding="utf-8")
        return WorkerResult(
            status="completed",
            summary="official risk audit",
            artifact_paths=["artifacts/risk_auditor/report.md"],
        )

    monkeypatch.setattr(rt, "run_worker", completed_risk_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    risk = TaskStore(run_dir).load_task("task-risk")
    assert risk.status is TaskStatus.completed, risk.error
    report = next(ref for ref in risk.artifact_refs if ref.run_relative_path.endswith("report.md"))
    assert report.artifact_type == "risk.audit_report"
    assert report.provenance_status == "passed"
    assert set(report.derived_from_artifact_ids) == {
        "provenance", "metrics-primary", "metrics-secondary", "trades", "equity", "config"
    }
    assert report.derived_from_manifest_generation == "generation-backtest"


def test_non_strict_risk_artifact_behavior_remains_compatible(monkeypatch, tmp_path: Path) -> None:
    runtime, run, _, run_dir = _runtime_case(tmp_path, strict=False, duplicate_metrics=False)

    def completed_risk_worker(**kwargs):
        report = run_dir / "artifacts" / "risk_auditor" / "report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("generic risk audit", encoding="utf-8")
        return WorkerResult(
            status="completed",
            summary="generic risk audit",
            artifact_paths=["artifacts/risk_auditor/report.md"],
        )

    monkeypatch.setattr(rt, "run_worker", completed_risk_worker)
    monkeypatch.setattr(runtime, "_prefetch_grounding_data", lambda _run: None)
    runtime._execute_run(run, threading.Event())

    risk = TaskStore(run_dir).load_task("task-risk")
    report = next(ref for ref in risk.artifact_refs if ref.run_relative_path.endswith("report.md"))
    assert report.artifact_type == "risk.audit_report"
