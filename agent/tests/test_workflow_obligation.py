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


class _BacktestTool(BaseTool):
    name = "backtest"
    description = "Run a backtest."
    parameters = {"type": "object", "properties": {"run_dir": {"type": "string"}}, "required": ["run_dir"]}

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: object) -> str:
        self.calls += 1
        return json.dumps({"status": "ok"})


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
