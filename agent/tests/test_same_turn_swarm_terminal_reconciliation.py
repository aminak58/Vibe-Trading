"""Same-turn canonical terminal reconciliation before the parent replies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.loop import AgentLoop
from src.agent.memory import WorkspaceMemory
from src.agent.tools import ToolRegistry
from src.agent.workflow_obligation import WorkflowObligationLedger, WorkflowStatus
from src.swarm.models import ArtifactRef, RunStatus, SwarmEvent, SwarmRun, SwarmTask
from src.swarm.store import SwarmStore


RUN_ID = "owned-same-turn"
LAUNCH_ID = "launch-same-turn"
SESSION_ID = "session-same-turn"
IDENTITY_HASH = "identity-same-turn"


def _terminal_run(status: RunStatus, *, session_id: str = SESSION_ID, launch_id: str = LAUNCH_ID) -> SwarmRun:
    refs = [
        ArtifactRef(
            artifact_id=f"artifact-{artifact_type}",
            producer_run_id=RUN_ID,
            producer_task_id="task-backtest",
            producer_agent_id="backtester",
            run_relative_path=f"artifacts/backtester/{artifact_type}.json",
            sha256="a" * 64,
            byte_size=1,
            execution_identity_hash=IDENTITY_HASH,
            artifact_type=artifact_type,
            provenance_status="passed",
        )
        for artifact_type in ("backtest.metrics", "backtest.trades", "backtest.equity")
    ]
    return SwarmRun(
        id=RUN_ID,
        preset_name="quant_scalp_desk",
        status=status,
        created_at="2026-09-06T00:00:00+00:00",
        completed_at="2026-09-06T00:01:00+00:00" if status is RunStatus.completed else None,
        final_report="official same-turn report" if status is RunStatus.completed else None,
        owner_session_id=session_id,
        launch_id=launch_id,
        identity_hash=IDENTITY_HASH,
        provenance_validation_status="passed",
        tasks=[
            SwarmTask(
                id="task-backtest",
                agent_id="backtester",
                prompt_template="x",
                status=(
                    "completed" if status is RunStatus.completed
                    else "in_progress" if status is RunStatus.running
                    else "failed"
                ),
                error="official backtest artifact gap" if status is RunStatus.failed else None,
                artifact_refs=refs,
            )
        ],
    )


def _waiting_agent(tmp_path: Path) -> AgentLoop:
    agent = AgentLoop(
        registry=ToolRegistry(), llm=SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path))
    )
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.bind_identity(IDENTITY_HASH)
    ledger.mark_swarm_started()
    ledger.bind_swarm_run(RUN_ID)
    ledger._obligation = ledger.obligation.transition(WorkflowStatus.WAITING, launch_id=LAUNCH_ID)
    ledger.persist()
    agent._workflow_obligation = ledger
    agent._trusted_owner_session_id = SESSION_ID
    return agent


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: SwarmRun) -> SwarmStore:
    root = tmp_path / "swarm-runs"
    store = SwarmStore(root)
    store.create_run(run)
    monkeypatch.setattr("src.swarm.store.swarm_runs_root", lambda: root)
    return store


def test_same_turn_completed_replaces_stale_parent_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _waiting_agent(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(RunStatus.completed))

    result = agent._reconcile_owned_swarm_before_final_response()

    assert result is not None
    assert result["reconciliation"]["status"] == "completed"
    assert result["content"] == "official same-turn report"
    assert agent._workflow_obligation.obligation.status is WorkflowStatus.COMPLETED


@pytest.mark.parametrize(
    ("status", "expected"),
    [(RunStatus.failed, "failed"), (RunStatus.cancelled, "cancelled")],
)
def test_same_turn_terminal_failure_replaces_stale_parent_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: RunStatus, expected: str
) -> None:
    agent = _waiting_agent(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(status))

    result = agent._reconcile_owned_swarm_before_final_response()

    assert result is not None
    assert result["reconciliation"]["status"] == expected
    assert f"terminal status: {expected}" in result["content"]
    assert "polling" not in result["content"].casefold()
    if status is RunStatus.failed:
        assert "official backtest artifact gap" in result["content"]
    assert agent._workflow_obligation.obligation.status is (
        WorkflowStatus.FAILED if status is RunStatus.failed else WorkflowStatus.CANCELLED
    )


def test_same_turn_running_preserves_polling_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _waiting_agent(tmp_path)
    store = _configure_store(monkeypatch, tmp_path, _terminal_run(RunStatus.running))
    store.append_event(
        RUN_ID,
        SwarmEvent(
            type="task_started",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )

    assert agent._reconcile_owned_swarm_before_final_response() is None
    assert agent._workflow_obligation.obligation.status is WorkflowStatus.WAITING


def test_same_turn_ownership_mismatch_cannot_replace_parent_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _waiting_agent(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(RunStatus.completed, launch_id="other-launch"))

    assert agent._reconcile_owned_swarm_before_final_response() is None
    assert agent._workflow_obligation.obligation.status is WorkflowStatus.WAITING


def test_same_turn_reconciliation_never_dispatches_or_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _waiting_agent(tmp_path)
    _configure_store(monkeypatch, tmp_path, _terminal_run(RunStatus.completed))
    monkeypatch.setattr(agent.registry, "execute", lambda *args, **kwargs: pytest.fail("must not dispatch"))

    assert agent._reconcile_owned_swarm_before_final_response() is not None


def test_same_turn_rejected_terminal_result_is_rendered_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _waiting_agent(tmp_path)
    monkeypatch.setattr(
        agent,
        "_reconcile_owned_swarm_from_store",
        lambda: {
            "status": "rejected",
            "run_id": RUN_ID,
            "terminal_reason": "preset_unavailable",
        },
    )

    result = agent._reconcile_owned_swarm_before_final_response()

    assert result is not None
    assert "terminal status: rejected" in result["content"]
    assert "preset_unavailable" in result["content"]


class _StaleTextResponse:
    content = "Polling exhausted; no backtest ran."
    tool_calls: list[object] = []
    reasoning_content = None
    has_tool_calls = False


class _StaleTextLLM:
    model_name = "test-model"

    def stream_chat(self, *args: object, **kwargs: object) -> _StaleTextResponse:
        return _StaleTextResponse()


def test_loop_replaces_stale_final_text_with_same_turn_official_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = AgentLoop(
        registry=ToolRegistry(),
        llm=_StaleTextLLM(),
        memory=WorkspaceMemory(run_dir=str(tmp_path)),
        max_iterations=1,
    )
    monkeypatch.setattr(
        agent,
        "_reconcile_owned_swarm_before_final_response",
        lambda: {
            "reconciliation": {"status": "completed", "run_id": RUN_ID},
            "content": "official same-turn report",
        },
    )

    result = agent.run("Summarize this.")

    assert result["content"] == "official same-turn report"
    assert result["swarm_reconciliation"] == {"status": "completed", "run_id": RUN_ID}
