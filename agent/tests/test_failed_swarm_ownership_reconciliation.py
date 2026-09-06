"""Terminal reconciliation for parent-owned Swarms without a live obligation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agent.loop import AgentLoop
from src.agent.memory import WorkspaceMemory
from src.agent.tools import ToolRegistry
from src.agent.workflow_obligation import WorkflowObligationLedger
from src.swarm.models import RunStatus, SwarmRun, SwarmTask
from src.swarm.store import SwarmStore


def _terminal_run(*, status: RunStatus, session_id: str = "session-1", launch_id: str | None = "launch-1") -> SwarmRun:
    return SwarmRun(
        id="owned-run",
        preset_name="quant_scalp_desk",
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        owner_session_id=session_id,
        launch_id=launch_id,
        tasks=[
            SwarmTask(
                id="task-backtest",
                agent_id="backtester",
                prompt_template="x",
                status="completed",
            )
        ],
    )


def _agent_with_running_ownership(
    tmp_path: Path,
    *,
    session_id: str = "session-1",
    launch_id: str | None = "launch-1",
) -> AgentLoop:
    ownership = {
        "status": "running",
        "run_id": "owned-run",
        "identity_hash": "identity-1",
        "launch_id": launch_id,
        "owner_session_id": session_id,
    }
    (tmp_path / "swarm_ownership.json").write_text(json.dumps(ownership), encoding="utf-8")
    agent = AgentLoop(
        registry=ToolRegistry(), llm=object(), memory=WorkspaceMemory(run_dir=str(tmp_path))
    )
    agent._trusted_owner_session_id = session_id
    agent._restore_swarm_ownership(tmp_path)
    agent._workflow_obligation = WorkflowObligationLedger(run_dir=tmp_path, user_message="continue")
    return agent


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: SwarmRun) -> None:
    store_root = tmp_path / "swarms"
    SwarmStore(store_root).create_run(run)
    monkeypatch.setattr("src.swarm.store.swarm_runs_root", lambda: store_root)


def test_owned_failed_swarm_reconciles_marker_to_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent_with_running_ownership(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(status=RunStatus.failed))

    result = agent._reconcile_owned_swarm_from_store()

    assert result is not None
    assert result["status"] == "failed"
    assert result["terminal_reason"] == "owned_swarm_terminal_failure"
    persisted = json.loads((tmp_path / "swarm_ownership.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["run_id"] == "owned-run"
    assert agent._active_swarm_run_id is None


def test_owned_cancelled_swarm_reconciles_marker_to_cancelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent_with_running_ownership(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(status=RunStatus.cancelled))

    result = agent._reconcile_owned_swarm_from_store()

    assert result is not None
    assert result["status"] == "cancelled"
    assert json.loads((tmp_path / "swarm_ownership.json").read_text())["status"] == "cancelled"


def test_owned_rejected_result_reconciles_marker_without_dispatch(tmp_path: Path) -> None:
    agent = _agent_with_running_ownership(tmp_path)

    agent._clear_completed_swarm_ownership(
        json.dumps({"run_id": "owned-run", "status": "rejected", "terminal_reason": "preset_unavailable"})
    )

    persisted = json.loads((tmp_path / "swarm_ownership.json").read_text())
    assert persisted["status"] == "rejected"
    assert persisted["terminal_reason"] == "preset_unavailable"
    assert agent._active_swarm_run_id is None


@pytest.mark.parametrize(
    "run_session,run_launch",
    [
        ("other-session", "launch-1"),
        ("session-1", "other-launch"),
    ],
)
def test_mismatched_owned_terminal_swarm_does_not_reconcile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_session: str, run_launch: str
) -> None:
    agent = _agent_with_running_ownership(tmp_path)
    _configure_store(
        monkeypatch,
        tmp_path,
        _terminal_run(status=RunStatus.failed, session_id=run_session, launch_id=run_launch),
    )

    assert agent._reconcile_owned_swarm_from_store() is None
    assert json.loads((tmp_path / "swarm_ownership.json").read_text())["status"] == "running"


def test_failed_owned_swarm_reports_partial_task_artifact_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent_with_running_ownership(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(status=RunStatus.failed))

    result = agent._reconcile_owned_swarm_from_store()

    assert result is not None
    assert result["artifact_status"]["task-backtest"]["status"] == "completed"
    assert result["artifact_status"]["task-backtest"]["artifact_count"] == 0


def test_missing_exact_owned_run_does_not_mutate_running_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _agent_with_running_ownership(tmp_path)
    _configure_store(
        monkeypatch,
        tmp_path,
        _terminal_run(status=RunStatus.failed).model_copy(update={"id": "another-run"}),
    )

    assert agent._reconcile_owned_swarm_from_store() is None
    assert json.loads((tmp_path / "swarm_ownership.json").read_text())["status"] == "running"
