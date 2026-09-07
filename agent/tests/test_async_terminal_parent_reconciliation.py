"""Async terminal Swarm finalization converges durable ownership views."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.workflow_obligation import WorkflowObligationLedger, WorkflowStatus
from src.swarm.models import ArtifactRef, RunStatus, SwarmRun, SwarmTask
from src.swarm.store import SwarmStore
from src.swarm.terminal_reconciliation import finalize_terminal_owned_swarm


SESSION = "trusted-session"
LAUNCH = "launch-owned"
SWARM = "owned-async-swarm"
IDENTITY = "identity-owned"
PARENT = "parent-owned"


def _terminal_run(status: RunStatus = RunStatus.failed) -> SwarmRun:
    refs = [
        ArtifactRef(
            artifact_id=f"artifact-{kind}",
            producer_run_id=SWARM,
            producer_task_id="task-backtest",
            producer_agent_id="backtester",
            run_relative_path=f"artifacts/backtester/{kind}.json",
            sha256="a" * 64,
            byte_size=1,
            execution_identity_hash=IDENTITY,
            artifact_type=kind,
            provenance_status="passed",
        )
        for kind in (
            "backtest.execution_provenance", "backtest.config", "backtest.run_card",
            "backtest.strategy", "backtest.metrics", "backtest.trades", "backtest.equity",
        )
    ]
    return SwarmRun(
        id=SWARM,
        preset_name="quant_scalp_desk",
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        owner_session_id=SESSION,
        launch_id=LAUNCH,
        identity_hash=IDENTITY,
        provenance_validation_status="passed",
        tasks=[
            SwarmTask(
                id="task-backtest",
                agent_id="backtester",
                prompt_template="x",
                status="completed",
                artifact_refs=refs,
            ),
            SwarmTask(
                id="task-risk",
                agent_id="risk_auditor",
                prompt_template="x",
                status="failed",
                error="provider_stream_error: 429 INFERENCE_CAP_ERROR",
            ),
            SwarmTask(
                id="task-report",
                agent_id="report_aggregator",
                prompt_template="x",
                status="blocked",
                error="task-risk=failed",
            ),
        ],
    )


def _owned_parent(tmp_path: Path) -> tuple[Path, Path, Path]:
    runs_dir = tmp_path / "runs"
    sessions_dir = tmp_path / "sessions"
    parent_dir = runs_dir / PARENT
    parent_dir.mkdir(parents=True)
    ledger = WorkflowObligationLedger(run_dir=parent_dir, user_message="Run Swarm")
    ledger.bind_identity(IDENTITY)
    ledger.mark_swarm_started()
    ledger.bind_swarm_run(SWARM)
    ledger._obligation = ledger.obligation.transition(WorkflowStatus.WAITING, launch_id=LAUNCH)
    ledger.persist()
    ownership = {
        "status": "running",
        "run_id": SWARM,
        "identity_hash": IDENTITY,
        "launch_id": LAUNCH,
        "owner_session_id": SESSION,
    }
    (parent_dir / "swarm_ownership.json").write_text(json.dumps(ownership), encoding="utf-8")
    index_dir = sessions_dir / SESSION
    index_dir.mkdir(parents=True)
    (index_dir / "swarm_ownership.json").write_text(
        json.dumps({
            "schema_version": "session-swarm-ownership/v1",
            "session_id": SESSION,
            "parent_run_id": PARENT,
            "swarm_run_id": SWARM,
            "identity_hash": IDENTITY,
            "launch_id": LAUNCH,
            "status": "running",
        }),
        encoding="utf-8",
    )
    return runs_dir, sessions_dir, parent_dir


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("status", "expected_obligation", "expected_index"),
    (
        (RunStatus.failed, "failed", "failed"),
        (RunStatus.completed, "completed", "completed"),
        (RunStatus.cancelled, "cancelled", "cancelled"),
    ),
)
def test_async_terminalization_converges_every_owned_view(
    tmp_path: Path, status: RunStatus, expected_obligation: str, expected_index: str
) -> None:
    """Removing the background finalizer must leave these views stale."""
    runs_dir, sessions_dir, parent_dir = _owned_parent(tmp_path)
    run = _terminal_run(status)
    if status is RunStatus.completed:
        run.tasks = [run.tasks[0]]

    result = finalize_terminal_owned_swarm(run, runs_dir=runs_dir, sessions_dir=sessions_dir)

    assert result is not None
    assert _read(parent_dir / "workflow_obligation.json")["status"] == expected_obligation
    assert _read(parent_dir / "swarm_ownership.json")["status"] == expected_index
    assert _read(sessions_dir / SESSION / "swarm_ownership.json")["status"] == expected_index


def test_async_provider_failure_preserves_reason_and_partial_artifact_status(tmp_path: Path) -> None:
    runs_dir, sessions_dir, parent_dir = _owned_parent(tmp_path)

    result = finalize_terminal_owned_swarm(
        _terminal_run(), runs_dir=runs_dir, sessions_dir=sessions_dir
    )

    assert result is not None
    assert "429 INFERENCE_CAP_ERROR" in str(result["terminal_reason"])
    assert result["artifact_status"]["task-backtest"]["status"] == "completed"
    assert result["official_backtest_artifact_types"] == [
        "backtest.config",
        "backtest.equity",
        "backtest.execution_provenance",
        "backtest.metrics",
        "backtest.run_card",
        "backtest.strategy",
        "backtest.trades",
    ]
    assert result["partial_backtest"]["completed"] is True
    index = _read(sessions_dir / SESSION / "swarm_ownership.json")
    assert "429 INFERENCE_CAP_ERROR" in str(index["terminal_reason"])
    assert _read(parent_dir / "workflow_obligation.json")["terminal_result_status"] == "failed"


def test_async_rejected_terminal_preserves_rejection_while_obligation_maps_to_failed(tmp_path: Path) -> None:
    runs_dir, sessions_dir, parent_dir = _owned_parent(tmp_path)
    base = _terminal_run()
    rejected = SimpleNamespace(
        id=base.id,
        status=SimpleNamespace(value="rejected"),
        owner_session_id=base.owner_session_id,
        launch_id=base.launch_id,
        tasks=base.tasks,
    )

    result = finalize_terminal_owned_swarm(rejected, runs_dir=runs_dir, sessions_dir=sessions_dir)

    assert result is not None and result["status"] == "failed"
    assert _read(parent_dir / "workflow_obligation.json")["status"] == "failed"
    assert _read(parent_dir / "swarm_ownership.json")["status"] == "rejected"
    assert _read(sessions_dir / SESSION / "swarm_ownership.json")["status"] == "rejected"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda run: run.model_copy(update={"owner_session_id": "other-session"}),
        lambda run: run.model_copy(update={"launch_id": "other-launch"}),
        lambda run: run.model_copy(update={"id": "other-swarm"}),
    ),
)
def test_async_terminalization_never_mutates_for_mismatched_ownership(
    tmp_path: Path, mutate
) -> None:
    runs_dir, sessions_dir, parent_dir = _owned_parent(tmp_path)

    assert finalize_terminal_owned_swarm(mutate(_terminal_run()), runs_dir=runs_dir, sessions_dir=sessions_dir) is None
    assert _read(parent_dir / "workflow_obligation.json")["status"] == "waiting"
    assert _read(parent_dir / "swarm_ownership.json")["status"] == "running"
    assert _read(sessions_dir / SESSION / "swarm_ownership.json")["status"] == "running"


def test_async_terminalization_is_idempotent_and_never_dispatches(tmp_path: Path) -> None:
    runs_dir, sessions_dir, parent_dir = _owned_parent(tmp_path)
    run = _terminal_run()

    first = finalize_terminal_owned_swarm(run, runs_dir=runs_dir, sessions_dir=sessions_dir)
    second = finalize_terminal_owned_swarm(run, runs_dir=runs_dir, sessions_dir=sessions_dir)

    assert first is not None
    assert second is not None
    assert _read(parent_dir / "swarm_ownership.json")["status"] == "failed"
    assert _read(sessions_dir / SESSION / "swarm_ownership.json")["status"] == "failed"


def test_async_terminalization_rejects_parent_path_escape(tmp_path: Path) -> None:
    runs_dir, sessions_dir, parent_dir = _owned_parent(tmp_path)
    index_path = sessions_dir / SESSION / "swarm_ownership.json"
    payload = _read(index_path)
    payload["parent_run_id"] = ".."
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    assert finalize_terminal_owned_swarm(_terminal_run(), runs_dir=runs_dir, sessions_dir=sessions_dir) is None
    assert _read(parent_dir / "workflow_obligation.json")["status"] == "waiting"


def test_runtime_invokes_server_owned_finalizer_after_canonical_terminal_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Removing the runtime call recreates the parent-return race."""
    import src.swarm.runtime as runtime_mod

    run = SwarmRun(
        id="runtime-terminal-run",
        preset_name="generic",
        status=RunStatus.pending,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store = SwarmStore(tmp_path / "swarm")
    store.create_run(run)
    runtime = runtime_mod.SwarmRuntime(store=store)
    finalized: list[str] = []
    monkeypatch.setattr(runtime_mod, "finalize_terminal_owned_swarm", lambda value: finalized.append(value.id))

    runtime._execute_run(run, threading.Event())

    assert store.load_run(run.id).status is RunStatus.completed
    assert finalized == [run.id]


def test_runtime_keeps_canonical_terminal_state_when_finalizer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupt ownership projection must not undo canonical terminalization."""
    import src.swarm.runtime as runtime_mod

    run = SwarmRun(
        id="runtime-finalizer-error",
        preset_name="generic",
        status=RunStatus.pending,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store = SwarmStore(tmp_path / "swarm")
    store.create_run(run)
    runtime = runtime_mod.SwarmRuntime(store=store)
    monkeypatch.setattr(
        runtime_mod,
        "finalize_terminal_owned_swarm",
        lambda _value: (_ for _ in ()).throw(OSError("ownership store unavailable")),
    )

    runtime._execute_run(run, threading.Event())

    assert store.load_run(run.id).status is RunStatus.completed
