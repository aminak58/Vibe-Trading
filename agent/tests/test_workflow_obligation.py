"""Server-side regression coverage for explicit Swarm workflow obligations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.context import ContextBuilder
from src.agent.loop import AgentLoop
from src.agent.memory import WorkspaceMemory
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter
from src.agent.workflow_obligation import (
    WorkflowObligationLedger,
    WorkflowStatus,
    obligation_from_current_user_message,
)


def test_current_user_swarm_directive_creates_required_obligation() -> None:
    obligation = obligation_from_current_user_message(
        "Run this with Swarm and use quant_scalp_desk."
    )
    assert obligation.mode.value == "swarm_required"
    assert obligation.required_preset == "quant_scalp_desk"


def test_historical_memory_prose_cannot_create_obligation() -> None:
    # The parser is intentionally called only with the current raw turn.
    obligation = obligation_from_current_user_message("Backtest this strategy.")
    assert obligation.mode.value == "none"


def test_required_preset_blocks_alternate_preset(tmp_path: Path) -> None:
    ledger = WorkflowObligationLedger(
        run_dir=tmp_path,
        user_message="Use quant_scalp_desk with Swarm.",
    )
    blocked = ledger.block("run_swarm", {"preset_name": "quant_strategy_desk"})
    assert blocked is not None
    assert json.loads(blocked)["error_code"] == "denied_by_orchestration_intent"
    assert ledger.block("run_swarm", {"preset_name": "quant_scalp_desk"}) is None


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
