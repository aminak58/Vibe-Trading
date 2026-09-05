"""Real-store coverage for launch-correlated pre-bind Swarm cancellation."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from src.agent.workflow_obligation import (
    WORKFLOW_OBLIGATION_ARTIFACT,
    WorkflowObligationLedger,
    WorkflowStatus,
)
from src.execution_identity import (
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionResolution,
    FallbackPolicy,
    SourceMode,
)
from src.swarm.models import RunStatus, SwarmRun, WorkerResult
from src.swarm.runtime import SwarmRuntime
from src.swarm.store import SwarmStore
from src.tools.swarm_tool import _cancel_launch_correlated_run_if_requested


def _strict_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-prebind",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
        ),
        requests=(ExecutionRequest(request_id="gold", symbol="XAUUSD", source="mt5"),),
        resolutions=(
            ExecutionResolution(
                request_id="gold",
                resolved_symbol="XAUUSD_o",
                source="mt5",
                asset_class="forex",
                market="forex",
                resolver_evidence_ref="resolver:prebind",
            ),
        ),
    )


def _generic_identity() -> ExecutionIdentity:
    return ExecutionIdentity(identity_id="generic-prebind", mode=ExecutionMode.GENERIC)


def _wait_for_terminal(store: SwarmStore, run_id: str) -> SwarmRun:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = store.load_run(run_id)
        if run is not None and run.status in {
            RunStatus.completed,
            RunStatus.failed,
            RunStatus.cancelled,
        }:
            return run
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach a terminal status")


def _hold_runtime_before_first_layer(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def pause_prefetch(self, run):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(SwarmRuntime, "_prefetch_grounding_data", pause_prefetch)
    return entered, release


def test_real_prebind_cancel_keeps_strict_run_cancelled_and_obligation_consumed(
    monkeypatch, tmp_path
):
    """A strict run cancelled before binding stays cancelled despite missing provenance."""
    entered, release = _hold_runtime_before_first_layer(monkeypatch)
    worker_calls: list[str] = []

    def no_network_worker(*args, **kwargs):
        worker_calls.append(kwargs["task"].id)
        return WorkerResult(status="completed", summary="controlled")

    monkeypatch.setattr("src.swarm.runtime.run_worker", no_network_worker)
    store = SwarmStore(base_dir=tmp_path / "swarms")
    runtime = SwarmRuntime(store=store, max_workers=1)
    ledger = WorkflowObligationLedger(run_dir=tmp_path / "agent", user_message="Run Swarm")
    ledger.mark_swarm_started()
    launch_id = ledger.obligation.launch_id
    assert launch_id is not None
    persisted_obligation = json.loads(
        (tmp_path / "agent" / WORKFLOW_OBLIGATION_ARTIFACT).read_text(encoding="utf-8")
    )
    assert persisted_obligation["launch_id"] == launch_id

    run = runtime.start_run(
        "quant_scalp_desk",
        {"goal": "controlled cancellation", "market": "forex"},
        execution_identity=_strict_identity(),
        owner_session_id="session-owner",
        launch_id=launch_id,
    )
    assert entered.wait(timeout=2)
    persisted = store.load_run(run.id)
    assert persisted is not None
    assert persisted.launch_id == launch_id
    assert persisted.owner_session_id == "session-owner"
    assert ledger.obligation.swarm_run_id is None

    parent_cancel = threading.Event()
    parent_cancel.set()
    _cancel_launch_correlated_run_if_requested(
        runtime, store, run.id, launch_id, "session-owner", parent_cancel
    )
    assert ledger.obligation.swarm_run_id is None
    ledger.bind_swarm_run(run.id)
    assert ledger.obligation.swarm_run_id == run.id

    release.set()
    terminal = _wait_for_terminal(store, run.id)

    assert terminal.status is RunStatus.cancelled
    assert terminal.provenance_validation_status == "provenance_conflict"
    assert terminal.final_report is not None
    assert "PROVENANCE_CONFLICT" in terminal.final_report
    assert worker_calls == []

    ledger.record_swarm_result(json.dumps({"run_id": run.id, "status": terminal.status.value}))
    assert ledger.obligation.status is WorkflowStatus.CANCELLED
    assert ledger.block("run_swarm", {}) is not None


def test_actual_store_lookup_does_not_cancel_wrong_session_or_no_match(monkeypatch, tmp_path):
    """A signalled parent cannot cancel a run through another session or a missing launch."""
    entered, release = _hold_runtime_before_first_layer(monkeypatch)

    def no_network_worker(*args, **kwargs):
        return WorkerResult(status="completed", summary="controlled")

    monkeypatch.setattr("src.swarm.runtime.run_worker", no_network_worker)
    store = SwarmStore(base_dir=tmp_path / "swarms")
    runtime = SwarmRuntime(store=store, max_workers=1)
    run = runtime.start_run(
        "quant_scalp_desk",
        {"goal": "negative cancellation", "market": "forex"},
        execution_identity=_generic_identity(),
        owner_session_id="session-owner",
        launch_id="launch-exact",
    )
    assert entered.wait(timeout=2)
    store.create_run(
        SwarmRun(
            id="unrelated-run",
            preset_name="quant_scalp_desk",
            created_at=datetime.now(timezone.utc).isoformat(),
            launch_id="launch-exact",
            owner_session_id="other-session",
        )
    )

    parent_cancel = threading.Event()
    parent_cancel.set()
    _cancel_launch_correlated_run_if_requested(
        runtime, store, run.id, "launch-exact", "other-session", parent_cancel
    )
    _cancel_launch_correlated_run_if_requested(
        runtime, store, run.id, "launch-missing", "session-owner", parent_cancel
    )

    release.set()
    terminal = _wait_for_terminal(store, run.id)
    unrelated = store.load_run("unrelated-run")
    assert terminal.status is not RunStatus.cancelled
    assert unrelated is not None and unrelated.status is RunStatus.pending
