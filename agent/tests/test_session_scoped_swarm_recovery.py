"""Recovery of an owned Swarm from a later UI parent run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.agent.loop as loop_mod
from src.agent.loop import AgentLoop
from src.agent.memory import WorkspaceMemory
from src.agent.tools import ToolRegistry
from src.agent.workflow_obligation import WorkflowObligationLedger, WorkflowStatus
from src.swarm.models import ArtifactRef, RunStatus, SwarmRun, SwarmTask, TaskStatus
from src.swarm.store import SwarmStore


SESSION = "trusted-session"
LAUNCH = "launch-owned"
SWARM = "owned-session-swarm"


def _owned_run(status: RunStatus, *, session: str = SESSION, launch: str = LAUNCH) -> SwarmRun:
    return SwarmRun(
        id=SWARM,
        preset_name="quant_scalp_desk",
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat() if status is not RunStatus.running else None,
        owner_session_id=session,
        launch_id=launch,
        tasks=[SwarmTask(id="task-backtest", agent_id="backtester", prompt_template="x", status="failed", error="official report blocked")],
    )


def _old_owner(tmp_path: Path) -> tuple[AgentLoop, Path]:
    old = tmp_path / "runs" / "old-parent"
    old.mkdir(parents=True)
    ledger = WorkflowObligationLedger(run_dir=old, user_message="Run Swarm")
    ledger.bind_identity("identity-1")
    ledger.mark_swarm_started()
    ledger.bind_swarm_run(SWARM)
    ledger._obligation = ledger.obligation.transition(WorkflowStatus.WAITING, launch_id=LAUNCH)
    ledger.persist()
    (old / "swarm_ownership.json").write_text(json.dumps({
        "status": "running", "run_id": SWARM, "identity_hash": "identity-1",
        "launch_id": LAUNCH, "owner_session_id": SESSION,
    }), encoding="utf-8")
    agent = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(old)))
    agent._trusted_owner_session_id = SESSION
    agent._active_swarm_run_id = SWARM
    agent._active_swarm_identity_hash = "identity-1"
    agent._active_swarm_launch_id = LAUNCH
    agent._active_swarm_owner_session_id = SESSION
    return agent, old


def _configure_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: SwarmRun) -> Path:
    sessions = tmp_path / "sessions"
    swarm_root = tmp_path / "swarm-runs"
    SwarmStore(swarm_root).create_run(run)
    monkeypatch.setattr(loop_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(loop_mod, "SESSIONS_DIR", sessions)
    monkeypatch.setattr("src.swarm.store.swarm_runs_root", lambda: swarm_root)
    return sessions


def test_new_parent_recovers_previous_failed_owned_swarm_via_session_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, old_dir = _old_owner(tmp_path)
    sessions = _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed))
    old_agent._persist_session_swarm_ownership("running")

    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION
    fresh._workflow_obligation = WorkflowObligationLedger(run_dir=Path(fresh.memory.run_dir), user_message="status only")
    result = fresh._reconcile_session_owned_swarm_from_store()

    assert result is not None and result["status"] == "failed"
    assert "official report blocked" in result["terminal_reason"]
    assert json.loads((old_dir / "workflow_obligation.json").read_text()) ["status"] == "failed"
    assert json.loads((old_dir / "swarm_ownership.json").read_text()) ["status"] == "failed"
    assert json.loads((sessions / SESSION / "swarm_ownership.json").read_text()) ["status"] == "failed"


def test_session_recovery_rejects_another_sessions_owned_swarm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, old_dir = _old_owner(tmp_path)
    _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed, session="other-session"))
    old_agent._persist_session_swarm_ownership("running")
    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION
    fresh._workflow_obligation = WorkflowObligationLedger(run_dir=Path(fresh.memory.run_dir), user_message="status only")

    assert fresh._reconcile_session_owned_swarm_from_store() is None
    assert json.loads((old_dir / "swarm_ownership.json").read_text()) ["status"] == "running"


def test_no_session_index_never_fabricates_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed))
    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION
    fresh._workflow_obligation = WorkflowObligationLedger(run_dir=Path(fresh.memory.run_dir), user_message="status only")

    assert fresh._reconcile_session_owned_swarm_from_store() is None


def test_session_recovery_preserves_a_genuinely_running_owned_swarm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, old_dir = _old_owner(tmp_path)
    run = _owned_run(RunStatus.running)
    run.tasks[0].status = TaskStatus.in_progress
    run.tasks[0].error = None
    _configure_roots(monkeypatch, tmp_path, run)
    old_agent._persist_session_swarm_ownership("running")

    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION

    assert fresh._reconcile_session_owned_swarm_from_store() == {"status": "running", "run_id": SWARM}
    assert json.loads((old_dir / "swarm_ownership.json").read_text())["status"] == "running"


def test_new_execution_is_blocked_while_session_index_owns_a_running_swarm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, _ = _old_owner(tmp_path)
    run = _owned_run(RunStatus.running)
    run.tasks[0].status = TaskStatus.in_progress
    run.tasks[0].error = None
    _configure_roots(monkeypatch, tmp_path, run)
    old_agent._persist_session_swarm_ownership("running")
    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION

    denial = fresh._swarm_ownership_block("run_swarm")

    assert denial is not None
    assert json.loads(denial)["run_id"] == SWARM


def test_official_start_registers_session_ownership_before_wait_budget_exhausts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, _ = _old_owner(tmp_path)
    sessions = _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.running))
    old_agent._workflow_obligation = WorkflowObligationLedger(
        run_dir=Path(old_agent.memory.run_dir), user_message=""
    )

    old_agent._bind_started_swarm_run(SWARM)

    index = json.loads((sessions / SESSION / "swarm_ownership.json").read_text())
    assert index["parent_run_id"] == "old-parent"
    assert index["swarm_run_id"] == SWARM
    assert index["launch_id"] == LAUNCH
    assert index["status"] == "running"


@pytest.mark.parametrize(
    ("status", "obligation_status"),
    (
        ("failed", "failed"),
        ("completed", "completed"),
        ("cancelled", "cancelled"),
        ("rejected", "failed"),
    ),
)
def test_terminal_run_swarm_result_converges_all_owned_state_views(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    obligation_status: str,
) -> None:
    """The direct tool result path must not leave its index running."""
    owner, old_dir = _old_owner(tmp_path)
    sessions = _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed))
    owner._workflow_obligation = WorkflowObligationLedger(run_dir=old_dir, user_message="continue")
    owner._persist_swarm_ownership("running")
    owner._persist_session_swarm_ownership("running")
    assert owner._workflow_obligation.obligation.mode.value == "swarm_required"
    assert owner._workflow_obligation.obligation.swarm_run_id == SWARM

    owner._record_owned_swarm_tool_result(
        json.dumps({"run_id": SWARM, "status": status, "terminal_reason": "terminal-test"})
    )

    assert json.loads((old_dir / "workflow_obligation.json").read_text())["status"] == obligation_status
    assert json.loads((old_dir / "swarm_ownership.json").read_text())["status"] == status
    index = json.loads((sessions / SESSION / "swarm_ownership.json").read_text())
    assert index["status"] == status
    assert index["terminal_reason"] == "terminal-test"
    assert owner._active_swarm_run_id is None


def test_mismatched_terminal_tool_result_does_not_mutate_owned_views(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner, old_dir = _old_owner(tmp_path)
    sessions = _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed))
    owner._workflow_obligation = WorkflowObligationLedger(run_dir=old_dir, user_message="continue")
    owner._persist_swarm_ownership("running")
    owner._persist_session_swarm_ownership("running")

    owner._record_owned_swarm_tool_result(
        json.dumps({"run_id": "other-swarm", "status": "failed"})
    )

    assert json.loads((old_dir / "workflow_obligation.json").read_text())["status"] == "waiting"
    assert json.loads((old_dir / "swarm_ownership.json").read_text())["status"] == "running"
    assert json.loads((sessions / SESSION / "swarm_ownership.json").read_text())["status"] == "running"


def test_repeated_terminal_tool_result_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner, old_dir = _old_owner(tmp_path)
    sessions = _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed))
    owner._workflow_obligation = WorkflowObligationLedger(run_dir=old_dir, user_message="continue")
    owner._persist_swarm_ownership("running")
    owner._persist_session_swarm_ownership("running")
    result = json.dumps({"run_id": SWARM, "status": "failed", "terminal_reason": "terminal-test"})

    owner._record_owned_swarm_tool_result(result)
    owner._record_owned_swarm_tool_result(result)

    assert json.loads((old_dir / "workflow_obligation.json").read_text())["status"] == "failed"
    assert json.loads((old_dir / "swarm_ownership.json").read_text())["status"] == "failed"
    assert json.loads((sessions / SESSION / "swarm_ownership.json").read_text())["status"] == "failed"


def test_repeated_session_recovery_returns_the_same_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, _ = _old_owner(tmp_path)
    _configure_roots(monkeypatch, tmp_path, _owned_run(RunStatus.failed))
    old_agent._persist_session_swarm_ownership("running")
    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION

    first = fresh._reconcile_session_owned_swarm_from_store()
    second = fresh._reconcile_session_owned_swarm_from_store()

    assert first is not None and second == first


def test_session_recovery_returns_completed_official_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old_agent, old_dir = _old_owner(tmp_path)
    refs = [
        ArtifactRef(
            artifact_id=f"artifact-{kind}", producer_run_id=SWARM,
            producer_task_id="task-backtest", producer_agent_id="backtester",
            run_relative_path=f"artifacts/backtester/{kind}.json", sha256="a" * 64,
            byte_size=1, execution_identity_hash="identity-1", artifact_type=kind,
            provenance_status="passed",
        )
        for kind in ("backtest.metrics", "backtest.trades", "backtest.equity")
    ]
    run = _owned_run(RunStatus.completed)
    run.identity_hash = "identity-1"
    run.provenance_validation_status = "passed"
    run.final_report = "official completed report"
    run.tasks[0].status = TaskStatus.completed
    run.tasks[0].artifact_refs = refs
    sessions = _configure_roots(monkeypatch, tmp_path, run)
    old_agent._persist_session_swarm_ownership("running")
    fresh = AgentLoop(ToolRegistry(), SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")))
    fresh._trusted_owner_session_id = SESSION

    result = fresh._reconcile_session_owned_swarm_from_store()

    assert result is not None and result["status"] == "completed"
    assert result["final_report"] == "official completed report"
    assert json.loads((old_dir / "workflow_obligation.json").read_text())["status"] == "completed"
    assert json.loads((sessions / SESSION / "swarm_ownership.json").read_text())["status"] == "completed"


class _StaleResponse:
    content = "Polling exhausted; no backtest ran."
    tool_calls: list[object] = []
    reasoning_content = None
    has_tool_calls = False


class _StaleLLM:
    model_name = "test-model"

    def stream_chat(self, *args: object, **kwargs: object) -> _StaleResponse:
        pytest.fail("a terminal session recovery must not invoke the model")


def test_new_ui_turn_replaces_model_stale_text_with_session_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = AgentLoop(
        ToolRegistry(), _StaleLLM(), memory=WorkspaceMemory(run_dir=str(tmp_path / "runs" / "new-parent")), max_iterations=1
    )
    monkeypatch.setattr(agent, "_reconcile_owned_swarm_from_store", lambda: None)
    monkeypatch.setattr(
        agent,
        "_reconcile_session_owned_swarm_from_store",
        lambda: {"status": "failed", "run_id": SWARM, "terminal_reason": "official report blocked"},
    )

    result = agent.run("Status only", session_id=SESSION)

    assert "terminal status: failed" in result["content"]
    assert "official report blocked" in result["content"]
    assert result["swarm_reconciliation"]["run_id"] == SWARM
