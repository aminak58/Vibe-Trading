"""Server-side regression coverage for explicit Swarm workflow obligations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.context import ContextBuilder
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
from src.agent.loop import AgentLoop
from src.agent.memory import WorkspaceMemory
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter
from src.agent.workflow_obligation import (
    PresetRequirementKind,
    WorkflowObligationLedger,
    WorkflowStatus,
    obligation_from_current_user_message,
)
from src.swarm.presets import resolve_source_scoped_preset
from src.swarm.models import ArtifactRef, RunStatus, SwarmRun


def _verified_mt5_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-gold",
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
                request_id="gold", resolved_symbol="XAUUSD_o", source="mt5",
                asset_class="forex", market="forex", resolver_evidence_ref="resolver:1",
            ),
        ),
    )


def test_current_user_swarm_directive_creates_required_obligation() -> None:
    obligation = obligation_from_current_user_message(
        "Run this with Swarm and use quant_scalp_desk."
    )
    assert obligation.mode.value == "swarm_required"
    assert obligation.required_preset == "quant_scalp_desk"
    assert obligation.requested_preset == "quant_scalp_desk"
    assert obligation.selected_preset == "quant_scalp_desk"
    assert obligation.selection_reason == "explicit_user_request"


def test_capability_swarm_directive_is_not_fabricated_as_exact_preset() -> None:
    obligation = obligation_from_current_user_message(
        "Use the strict-capable Swarm preset and preserve provenance through Swarm, backtest, and final reporting."
    )
    assert obligation.mode.value == "swarm_required"
    assert obligation.preset_requirement.kind is PresetRequirementKind.CAPABILITY
    assert obligation.preset_requirement.source_scoped_execution is True
    assert obligation.requested_preset is None
    assert obligation.selected_preset is None


def test_historical_memory_prose_cannot_create_obligation() -> None:
    # The parser is intentionally called only with the current raw turn.
    obligation = obligation_from_current_user_message("Backtest this strategy.")
    assert obligation.mode.value == "none"


@pytest.mark.parametrize(
    "message",
    [
        "Do not use Swarm for this backtest.",
        "Complete this without Swarm.",
        "Should we use Swarm?",
        "Swarm may be useful, but first explain the plan.",
        "Compare this to the previous Swarm run.",
        'A historical note said "run Swarm" yesterday.',
        "`run Swarm` was an old example.",
        "> run Swarm\nHistorical quoted instruction only.",
    ],
)
def test_non_directive_or_negated_swarm_prose_creates_no_obligation(message: str) -> None:
    assert obligation_from_current_user_message(message).mode.value == "none"


def test_required_preset_blocks_alternate_preset(tmp_path: Path) -> None:
    ledger = WorkflowObligationLedger(
        run_dir=tmp_path,
        user_message="Use quant_scalp_desk with Swarm.",
    )
    blocked = ledger.block("run_swarm", {"preset_name": "quant_strategy_desk"})
    assert blocked is not None
    assert json.loads(blocked)["error_code"] == "denied_by_orchestration_intent"
    assert ledger.block("run_swarm", {"preset_name": "quant_scalp_desk"}) is None


def test_capability_requirement_selects_only_verified_compatible_preset(tmp_path: Path) -> None:
    ledger = WorkflowObligationLedger(
        run_dir=tmp_path,
        user_message="Use the strict-capable Swarm preset.",
    )
    arguments: dict[str, object] = {"prompt": "XAUUSD source=mt5"}
    assert ledger.prepare_run_swarm(arguments, _verified_mt5_identity()) is None
    assert arguments["preset_name"] == "quant_scalp_desk"
    assert ledger.obligation.requested_preset is None
    assert ledger.obligation.selected_preset == "quant_scalp_desk"
    assert ledger.obligation.selection_reason == "capability_match"


def test_capability_resolution_blocks_when_no_preset_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.swarm.presets._preset_search_dirs", lambda: (tmp_path,))
    preset, error = resolve_source_scoped_preset(_verified_mt5_identity(), require_synthetic_forbid=True)
    assert preset is None
    assert error is not None and "No installed Swarm preset" in error


def test_capability_resolution_refuses_ambiguous_presets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = (
        "capabilities:\n"
        "  asset_classes: [forex]\n"
        "  sources: [mt5]\n"
        "  source_scoped_execution: true\n"
        "  synthetic_data: forbid\n"
    )
    (tmp_path / "first.yaml").write_text(content, encoding="utf-8")
    (tmp_path / "second.yaml").write_text(content, encoding="utf-8")
    monkeypatch.setattr("src.swarm.presets._preset_search_dirs", lambda: (tmp_path,))
    preset, error = resolve_source_scoped_preset(_verified_mt5_identity(), require_synthetic_forbid=True)
    assert preset is None
    assert error is not None and "More than one installed Swarm preset" in error


def test_required_swarm_blocks_direct_backtest_but_allows_prerequisites(tmp_path: Path) -> None:
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm for XAUUSD")
    assert ledger.block("backtest", {"run_dir": str(tmp_path)}) is not None
    assert ledger.block("search_symbol", {"query": "XAUUSD", "source": "mt5"}) is None
    assert ledger.block("get_backtest_contract", {"source": "mt5"}) is None


def test_waiting_obligation_restores_after_restart(tmp_path: Path) -> None:
    first = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    first.bind_identity("identity-hash")
    first.record_swarm_result(json.dumps({"status": "running", "run_id": "swarm-1", "wait_budget_exhausted": True}))
    restored = WorkflowObligationLedger(run_dir=tmp_path, user_message="continue")
    assert restored.obligation.status is WorkflowStatus.WAITING
    assert restored.obligation.swarm_run_id == "swarm-1"
    assert restored.obligation.identity_hash == "identity-hash"


def test_v1_exact_preset_obligation_migrates_without_losing_constraint(tmp_path: Path) -> None:
    (tmp_path / "workflow_obligation.json").write_text(
        json.dumps(
            {
                "schema_version": "workflow-obligation/v1",
                "mode": "swarm_required",
                "required_preset": "quant_scalp_desk",
                "status": "waiting",
                "authority": "current_user",
            }
        ),
        encoding="utf-8",
    )
    restored = WorkflowObligationLedger(run_dir=tmp_path, user_message="continue")
    assert restored.obligation.requested_preset == "quant_scalp_desk"
    assert restored.obligation.selected_preset == "quant_scalp_desk"
    assert restored.obligation.selection_reason == "legacy_v1_exact_preset"


def test_failed_swarm_cannot_silently_downgrade_to_direct_backtest(tmp_path: Path) -> None:
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.record_swarm_result(json.dumps({"status": "failed", "run_id": "swarm-1"}))
    assert ledger.obligation.status is WorkflowStatus.FAILED
    assert ledger.block("backtest", {"run_dir": str(tmp_path)}) is not None


@pytest.mark.parametrize("status", ["failed", "completed", "cancelled"])
def test_terminal_swarm_consumes_current_objective_attempt(tmp_path: Path, status: str) -> None:
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.mark_swarm_started()
    ledger.record_swarm_result(json.dumps({"status": status, "run_id": "swarm-1"}))
    assert ledger.block("run_swarm", {}) is not None
    assert ledger.block("backtest", {}) is not None


def test_launch_is_persisted_before_run_id_binding(tmp_path: Path) -> None:
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.mark_swarm_started()
    assert ledger.obligation.launch_id and ledger.obligation.swarm_run_id is None
    # Cancellation before bind cannot name or cancel an unrelated run and the
    # one-attempt gate remains fail-closed.
    assert ledger.block("run_swarm", {}) is not None


class _BacktestTool(BaseTool):
    name = "backtest"
    description = "Run a backtest."
    parameters = {"type": "object", "properties": {"run_dir": {"type": "string"}}, "required": ["run_dir"]}

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: object) -> str:
        self.calls += 1
        return json.dumps({"status": "ok"})


@pytest.mark.parametrize("payload", [
    {"status": "error", "error_code": "denied_by_orchestration_intent"},
    {"status": "failed", "run_id": None},
    {"status": "completed", "run_id": "another-run"},
])
def test_unowned_results_preserve_bound_obligation(tmp_path, payload):
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.mark_swarm_started()
    ledger.bind_swarm_run("owned-run")
    before = ledger.obligation.model_dump()
    ledger.record_swarm_result(json.dumps(payload))
    assert ledger.obligation.model_dump() == before
    persisted = json.loads((tmp_path / "workflow_obligation.json").read_text())
    assert persisted["swarm_run_id"] == "owned-run"


def test_duplicate_mark_started_cannot_replace_launch(tmp_path):
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.mark_swarm_started()
    ledger.bind_swarm_run("owned-run")
    before = ledger.obligation.model_dump()
    ledger.mark_swarm_started()
    assert ledger.obligation.model_dump() == before


def test_loop_rejected_swarm_does_not_enter_ownership_mutation(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    registry = ToolRegistry()
    agent = AgentLoop(registry=registry, llm=SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(run_dir)))
    ledger = WorkflowObligationLedger(run_dir=run_dir, user_message="Run Swarm")
    ledger.mark_swarm_started()
    ledger.bind_swarm_run("owned-run")
    ledger.record_swarm_result(json.dumps({"status": "failed", "run_id": "owned-run"}))
    agent._workflow_obligation = ledger
    before = ledger.obligation.model_dump()
    def forbidden_mutation(*args, **kwargs):
        pytest.fail("Rejected invocation entered ownership mutation")
    monkeypatch.setattr(ledger, "record_swarm_result", forbidden_mutation)
    monkeypatch.setattr(agent, "_record_swarm_ownership", forbidden_mutation)
    trace = TraceWriter(run_dir)
    messages = []
    try:
        agent._process_tool_calls(
            [SimpleNamespace(id="retry-denied", name="run_swarm", arguments={})],
            ContextBuilder, messages, trace, [], 1,
        )
    finally:
        trace.close()
    assert any("denied_by_orchestration_intent" in m["content"] for m in messages)
    assert ledger.obligation.model_dump() == before


def test_loop_blocks_direct_backtest_before_tool_dispatch(tmp_path: Path) -> None:
    tool = _BacktestTool()
    registry = ToolRegistry()
    registry.register(tool)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = AgentLoop(registry=registry, llm=SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(run_dir)))
    agent._workflow_obligation = WorkflowObligationLedger(run_dir=run_dir, user_message="Run Swarm")
    trace = TraceWriter(run_dir)
    messages: list[dict] = []
    agent._process_tool_calls(
        [SimpleNamespace(id="direct", name="backtest", arguments={"run_dir": str(run_dir)})],
        ContextBuilder,
        messages,
        trace,
        [],
        1,
    )
    trace.close()
    assert tool.calls == 0
    assert any("denied_by_orchestration_intent" in message["content"] for message in messages)


def _official_terminal_run(*, run_id: str = "swarm-20260906-045503-2373cf2c", status: RunStatus = RunStatus.completed,
                           provenance: str = "passed", session_id: str = "session-1", launch_id: str = "launch-1",
                           artifact_types: tuple[str, ...] = (
                               "backtest.metrics", "backtest.trades", "backtest.equity",
                           )) -> SwarmRun:
    refs = [
        ArtifactRef(
            artifact_id=f"artifact-{artifact_type}",
            producer_run_id=run_id,
            producer_task_id="task-backtest",
            producer_agent_id="backtester",
            run_relative_path=f"artifacts/backtester/{artifact_type}.json",
            sha256="a" * 64,
            byte_size=1,
            execution_identity_hash="identity-1",
            artifact_type=artifact_type,
            provenance_status="passed",
        )
        for artifact_type in artifact_types
    ]
    from src.swarm.models import SwarmTask
    return SwarmRun(
        id=run_id,
        preset_name="quant_scalp_desk",
        status=status,
        created_at="2026-09-06T04:55:03+00:00",
        completed_at="2026-09-06T05:20:18+00:00" if status is RunStatus.completed else None,
        final_report="official final report" if status is RunStatus.completed else None,
        owner_session_id=session_id,
        launch_id=launch_id,
        identity_hash="identity-1",
        provenance_validation_status=provenance,
        tasks=[SwarmTask(id="task-backtest", agent_id="backtester", prompt_template="x", artifact_refs=refs)],
    )


def _waiting_owned_ledger(tmp_path: Path) -> WorkflowObligationLedger:
    ledger = WorkflowObligationLedger(run_dir=tmp_path, user_message="Run Swarm")
    ledger.bind_identity("identity-1")
    ledger.mark_swarm_started()
    ledger.bind_swarm_run("swarm-20260906-045503-2373cf2c")
    ledger._obligation = ledger.obligation.transition(WorkflowStatus.WAITING, launch_id="launch-1")
    ledger.persist()
    return ledger


def test_waiting_owned_completed_swarm_reconciles_from_official_artifacts(tmp_path: Path) -> None:
    ledger = _waiting_owned_ledger(tmp_path)
    result = ledger.reconcile_owned_swarm(_official_terminal_run(), owner_session_id="session-1")
    assert result is not None
    assert result["status"] == "completed"
    assert result["final_report"] == "official final report"
    assert set(result["artifact_ids"]) == {
        "artifact-backtest.metrics", "artifact-backtest.trades", "artifact-backtest.equity",
    }
    assert ledger.obligation.status is WorkflowStatus.COMPLETED
    assert ledger.obligation.dispatch_attempted is True
    assert ledger.block("run_swarm", {}) is not None


def test_completed_owned_swarm_with_missing_official_artifacts_is_terminal_gap(tmp_path: Path) -> None:
    ledger = _waiting_owned_ledger(tmp_path)
    result = ledger.reconcile_owned_swarm(
        _official_terminal_run(artifact_types=("backtest.metrics",)), owner_session_id="session-1"
    )
    assert result is not None
    assert result["status"] == "completed_with_artifact_gap"
    assert result["missing_artifact_types"] == ["backtest.equity", "backtest.trades"]
    assert ledger.obligation.status is WorkflowStatus.FAILED
    assert ledger.obligation.terminal_reason == "completed_with_artifact_gap"


def test_late_terminal_failure_reconciles_actual_swarm_state(tmp_path: Path) -> None:
    ledger = _waiting_owned_ledger(tmp_path)
    result = ledger.reconcile_owned_swarm(
        _official_terminal_run(status=RunStatus.failed), owner_session_id="session-1"
    )
    assert result is not None
    assert result["status"] == "failed"
    assert result["run_id"] == "swarm-20260906-045503-2373cf2c"
    assert result["terminal_reason"] == "owned_swarm_terminal_failure"
    assert result["artifact_status"]["task-backtest"]["status"] == "pending"
    assert ledger.obligation.status is WorkflowStatus.FAILED


def test_terminal_run_with_mismatched_ownership_cannot_reconcile_obligation(tmp_path: Path) -> None:
    ledger = _waiting_owned_ledger(tmp_path)
    result = ledger.reconcile_owned_swarm(_official_terminal_run(launch_id="other-launch"), owner_session_id="session-1")
    assert result is None
    assert ledger.obligation.status is WorkflowStatus.WAITING


def test_terminal_run_with_mismatched_session_cannot_reconcile_obligation(tmp_path: Path) -> None:
    ledger = _waiting_owned_ledger(tmp_path)
    result = ledger.reconcile_owned_swarm(_official_terminal_run(), owner_session_id="other-session")
    assert result is None
    assert ledger.obligation.status is WorkflowStatus.WAITING


def test_loop_recovers_late_owned_swarm_completion_from_persisted_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.swarm.store import SwarmStore

    ledger = _waiting_owned_ledger(tmp_path / "parent")
    store = SwarmStore(tmp_path / "swarm-runs")
    store.create_run(_official_terminal_run())
    agent = AgentLoop(
        registry=ToolRegistry(), llm=SimpleNamespace(), memory=WorkspaceMemory(run_dir=str(tmp_path / "parent"))
    )
    agent._workflow_obligation = ledger
    agent._trusted_owner_session_id = "session-1"
    monkeypatch.setattr("src.swarm.store.swarm_runs_root", lambda: tmp_path / "swarm-runs")

    result = agent._reconcile_owned_swarm_from_store()

    assert result is not None
    assert result["status"] == "completed"
    assert result["final_report"] == "official final report"
    assert ledger.obligation.status is WorkflowStatus.COMPLETED
