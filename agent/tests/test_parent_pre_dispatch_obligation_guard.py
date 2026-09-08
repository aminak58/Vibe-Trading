"""TDD regressions for strict Swarm obligations before dispatch.

A current-user SWARM_REQUIRED request is not complete merely because the
parent model produced prose.  These tests keep the pre-dispatch lifecycle
separate from the Swarm runtime itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.agent.loop import AgentLoop
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter
from src.agent.context import ContextBuilder
from src.agent.workflow_obligation import WorkflowObligationLedger


STRICT_SWARM_REQUEST = (
    "لطفا با استفاده از Swarm این استراتژی XAUUSD را با داده MT5 ارزیابی کن."
)
STRICT_SWARM_REQUEST_WITH_ATTACHMENT = (
    "[Uploaded file: strategy.pdf, path: uploads/strategy.pdf]\n\n"
    + STRICT_SWARM_REQUEST
)


class _Response:
    content: str = ""
    tool_calls: list[Any]
    reasoning_content: str | None = None

    def __init__(self, *, content: str = "", tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.has_tool_calls = bool(self.tool_calls)


class _FinalTextLLM:
    def stream_chat(self, messages, tools=None, **kwargs):
        return _Response(content="I am stopping before dispatch.")

    def chat(self, messages, **kwargs):
        return _Response()


class _RepeatedProbeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def stream_chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if tools is None:
            return _Response(content="forced final text")
        return _Response(
            tool_calls=[
                SimpleNamespace(
                    id=f"probe-{self.calls}", name="pre_dispatch_probe", arguments={}
                )
            ]
        )

    def chat(self, messages, **kwargs):
        return _Response()


class _ProbeTool(BaseTool):
    name = "pre_dispatch_probe"
    description = "test-only no-op"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        return json.dumps({"status": "ok"})


class _NonrepeatableDeterministicTool(BaseTool):
    name = "deterministic_contract"
    description = "test-only deterministic contract"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    deterministic = True

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return json.dumps({"status": "ok", "contract": "mt5", "calls": self.calls})


class _NonrepeatableUncachedTool(BaseTool):
    name = "uncached_contract"
    description = "test-only non-replayable contract"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return json.dumps({"status": "ok", "contract": "one-time"})


class _StrictMt5ResolverTool(BaseTool):
    """Resolve the one broker-native identity used by dispatch-priority tests."""

    name = "search_symbol"
    description = "test-only MT5 resolver"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return json.dumps(
            {
                "status": "ok",
                "data": {
                    "candidates": [
                        {
                            "requested_symbol": "XAUUSD",
                            "resolved_symbol": "XAUUSD_o",
                            "source": "mt5",
                            "source_namespace": "connected_mt5_broker",
                            "exchange": "MT5",
                            "type": "forex",
                            "market_type": "forex",
                        }
                    ]
                },
            }
        )


class _FailingStrictMt5ResolverTool(BaseTool):
    """Represent a resolver implementation that ran and returned an error."""

    name = "search_symbol"
    description = "test-only failing MT5 resolver"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return json.dumps(
            {
                "status": "error",
                "error_code": "resolver_unavailable",
                "message": "resolver implementation failed",
            }
        )


class _RecordingSwarmTool(BaseTool):
    name = "run_swarm"
    description = "test-only Swarm dispatcher"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    is_readonly = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        kwargs["__on_swarm_started"]("swarm-priority-test")
        return json.dumps(
            {
                "status": "failed",
                "run_id": "swarm-priority-test",
                "terminal_reason": "test_terminal_failure",
            }
        )


class _TerminalSwarmTool(BaseTool):
    """Return an owned terminal Swarm result after the dispatcher binds it."""

    name = "run_swarm"
    description = "test-only terminal Swarm dispatcher"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    is_readonly = False

    def __init__(self, payload: dict[str, Any], *, started_run_id: str | None = None) -> None:
        self.payload = payload
        self.started_run_id = started_run_id or str(payload["run_id"])
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        kwargs["__on_swarm_started"](self.started_run_id)
        return json.dumps(self.payload)


class _RunningSwarmTool(BaseTool):
    """Return an owned Swarm that outlives the direct wait budget."""

    name = "run_swarm"
    description = "test-only running Swarm dispatcher"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    is_readonly = False

    def __init__(self, *, run_id: str = "swarm-running") -> None:
        self.run_id = run_id
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        kwargs["__on_swarm_started"](self.run_id)
        return json.dumps(
            {
                "status": "running",
                "run_id": self.run_id,
                "wait_budget_exhausted": True,
            }
        )


class _RecordingAdvisoryTool(BaseTool):
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "test-only advisory tool"
        self.parameters = {"type": "object", "properties": {}}
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return json.dumps({"status": "ok"})


class _UnavailableDocumentTool(BaseTool):
    name = "read_document"
    description = "test-only unavailable attachment"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return json.dumps({"status": "error", "error": "attachment unavailable"})


class _ResolveThenAdvisoryLLM:
    """Models the observed drift: resolve first, then request advisory work."""

    def __init__(self, advisory_tool: str) -> None:
        self.advisory_tool = advisory_tool
        self.calls = 0

    def stream_chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if tools is None:
            return _Response(content="forced final text")
        if self.calls == 1:
            return _Response(
                tool_calls=[
                    SimpleNamespace(
                        id="resolve-mt5",
                        name="search_symbol",
                        arguments={"query": "XAUUSD", "source": "mt5"},
                    )
                ]
            )
        return _Response(
            tool_calls=[
                SimpleNamespace(
                    id=f"advisory-{self.calls}",
                    name=self.advisory_tool,
                    arguments={},
                )
            ]
        )

    def chat(self, messages, **kwargs):
        return _Response()


class _WrongAliasThenStopLLM:
    """Request a resolved alias where the strict resolver requires the request symbol."""

    def __init__(self) -> None:
        self.calls = 0

    def stream_chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if tools is None or self.calls > 1:
            return _Response(content="Stop after the denied resolver attempt.")
        return _Response(
            tool_calls=[
                SimpleNamespace(
                    id="wrong-resolver-alias",
                    name="search_symbol",
                    arguments={"query": "XAUUSD_o", "source": "mt5"},
                )
            ]
        )

    def chat(self, messages, **kwargs):
        return _Response()


class _AdvisoryAndResolveLLM:
    """Attempts to hide a skill load beside the one allowed resolver call."""

    def __init__(self, advisory_tool: str) -> None:
        self.advisory_tool = advisory_tool
        self.calls = 0

    def stream_chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if tools is None:
            return _Response(content="forced final text")
        return _Response(
            tool_calls=[
                SimpleNamespace(
                    id="advisory-before-identity",
                    name=self.advisory_tool,
                    arguments={},
                ),
                SimpleNamespace(
                    id="resolve-mt5",
                    name="search_symbol",
                    arguments={"query": "XAUUSD", "source": "mt5"},
                ),
            ]
        )

    def chat(self, messages, **kwargs):
        return _Response()


class _ResolveWithUnavailableAttachmentLLM:
    def __init__(self) -> None:
        self.calls = 0

    def stream_chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if tools is None:
            return _Response(content="forced final text")
        return _Response(
            tool_calls=[
                SimpleNamespace(
                    id="resolve-mt5",
                    name="search_symbol",
                    arguments={"query": "XAUUSD", "source": "mt5"},
                ),
                SimpleNamespace(
                    id="missing-pdf",
                    name="read_document",
                    arguments={"file_path": "uploads/missing.pdf"},
                ),
            ]
        )

    def chat(self, messages, **kwargs):
        return _Response()


def _agent(tmp_path: Path, llm: Any, registry: ToolRegistry | None = None, *, max_iterations: int = 8) -> AgentLoop:
    agent = AgentLoop(
        registry=registry or ToolRegistry(),
        llm=llm,
        max_iterations=max_iterations,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    return agent


def test_pending_swarm_obligation_cannot_finish_success_from_final_prose(tmp_path: Path) -> None:
    agent = _agent(tmp_path, _FinalTextLLM())

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    state = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    obligation = json.loads(
        (tmp_path / "run" / "workflow_obligation.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "failed"
    assert result["reason"] == "swarm_required_pre_dispatch_incomplete"
    assert state["status"] == "failed"
    assert obligation["mode"] == "swarm_required"
    assert obligation["dispatch_attempted"] is False
    assert obligation["swarm_run_id"] is None


def test_invalid_parent_config_blocks_undispatched_swarm_success(tmp_path: Path) -> None:
    agent = _agent(tmp_path, _FinalTextLLM())
    (tmp_path / "run" / "config.json").write_text(
        '{"source":"mt5", "end_date 2026-09-07"', encoding="utf-8"
    )

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert result["status"] == "failed"
    assert result["reason"] == "invalid_prepared_config"
    assert json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_deterministic_nonrepeatable_tool_replays_cached_payload_before_name_skip(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = _NonrepeatableDeterministicTool()
    registry.register(tool)
    agent = _agent(tmp_path, _FinalTextLLM(), registry)
    trace = TraceWriter(tmp_path / "run")
    messages: list[dict[str, Any]] = []
    try:
        for call_id in ("first", "after-cleared"):
            agent._process_tool_calls(
                [SimpleNamespace(id=call_id, name=tool.name, arguments={})],
                ContextBuilder,
                messages,
                trace,
                [],
                1,
            )
    finally:
        trace.close()

    assert tool.calls == 1
    assert json.loads(messages[1]["content"])["contract"] == "mt5"
    assert "skipped" not in messages[1]["content"]
    assert any(
        event["type"] == "tool_result_cached"
        for event in TraceWriter.read(tmp_path / "run")
    )


def test_cleared_nonreplayable_result_becomes_unavailable_not_satisfied(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = _NonrepeatableUncachedTool()
    registry.register(tool)
    agent = _agent(tmp_path, _FinalTextLLM(), registry)
    trace = TraceWriter(tmp_path / "run")
    messages: list[dict[str, Any]] = []
    try:
        agent._process_tool_calls(
            [SimpleNamespace(id="first", name=tool.name, arguments={})],
            ContextBuilder,
            messages,
            trace,
            [],
            1,
        )
        messages[0]["content"] = "[cleared]"
        agent._process_tool_calls(
            [SimpleNamespace(id="after-cleared", name=tool.name, arguments={})],
            ContextBuilder,
            messages,
            trace,
            [],
            2,
        )
    finally:
        trace.close()

    payload = json.loads(messages[-1]["content"])
    assert payload["error_code"] == "tool_result_unavailable"
    assert agent._pre_dispatch_tool_result_unavailable is True


def test_undispatched_swarm_stops_before_general_iteration_budget(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(_ProbeTool())
    llm = _RepeatedProbeLLM()
    agent = _agent(tmp_path, llm, registry, max_iterations=20)

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert result["status"] == "failed"
    assert result["reason"] == "swarm_required_pre_dispatch_budget_exhausted"
    assert result["iterations"] < 20
    assert llm.calls < 20


def test_completed_owned_swarm_remains_eligible_for_normal_parent_success(tmp_path: Path) -> None:
    agent = _agent(tmp_path, _FinalTextLLM())
    ledger = WorkflowObligationLedger(
        run_dir=tmp_path / "run", user_message=STRICT_SWARM_REQUEST
    )
    ledger.mark_swarm_started()
    ledger.bind_swarm_run("swarm-completed")
    ledger.record_swarm_result(
        json.dumps({"status": "completed", "run_id": "swarm-completed"})
    )
    agent._workflow_obligation = ledger

    assert agent._swarm_obligation_terminal_blocker(tmp_path / "run", []) is None


def test_verified_strict_swarm_dispatches_before_advisory_market_data(
    tmp_path: Path, monkeypatch
) -> None:
    """A model cannot spend its post-identity turn on a coverage probe."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _RecordingSwarmTool()
    advisory = _RecordingAdvisoryTool("get_market_data")
    registry.register(swarm)
    registry.register(advisory)
    llm = _ResolveThenAdvisoryLLM("get_market_data")
    agent = _agent(tmp_path, llm, registry, max_iterations=5)
    monkeypatch.setattr(
        WorkflowObligationLedger,
        "prepare_run_swarm",
        lambda _self, arguments, _identity: arguments.update(
            {"preset_name": "quant_scalp_desk"}
        ) or None,
    )

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert len(swarm.calls) == 1
    calls = [
        event["tool"]
        for event in TraceWriter.read(tmp_path / "run")
        if event.get("type") == "tool_call"
    ]
    assert calls == ["search_symbol", "run_swarm"]
    assert advisory.calls == 0
    assert swarm.calls[0]["prompt"] == STRICT_SWARM_REQUEST
    assert swarm.calls[0]["__execution_identity"].resolutions[0].resolved_symbol == "XAUUSD_o"
    assert result["status"] == "failed"


def test_authorization_blocked_resolver_does_not_reject_pending_identity(
    tmp_path: Path,
) -> None:
    """A resolver implementation that never ran cannot reject strict identity."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    agent = _agent(tmp_path, _WrongAliasThenStopLLM(), registry, max_iterations=3)

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    identity = json.loads(
        (tmp_path / "run" / "execution_identity.json").read_text(encoding="utf-8")
    )
    assert identity["status"] == "partially_specified"
    assert identity["resolutions"] == []
    assert result["reason"] == "swarm_required_pre_dispatch_incomplete"

    tool_result = next(
        event
        for event in TraceWriter.read(tmp_path / "run")
        if event.get("type") == "tool_result"
        and event.get("call_id") == "wrong-resolver-alias"
    )
    payload = json.loads(tool_result["result"])
    assert payload["error_code"] == "denied_by_execution_identity"
    assert payload["executed"] is False
    assert payload["authoritative_result"] is False
    assert payload["block_stage"] == "authorization"


def test_executed_resolver_failure_still_rejects_pending_identity(
    tmp_path: Path,
) -> None:
    """The non-mutation rule must not hide failures from a resolver that ran."""
    registry = ToolRegistry()
    registry.register(_FailingStrictMt5ResolverTool())
    agent = _agent(tmp_path, _ResolveThenAdvisoryLLM("get_market_data"), registry)

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    identity = json.loads(
        (tmp_path / "run" / "execution_identity.json").read_text(encoding="utf-8")
    )
    assert identity["status"] == "rejected"
    assert identity["resolutions"] == []
    assert result["reason"] == "strict_execution_identity_unavailable"


def test_owned_running_swarm_quiesces_parent_before_advisory_tools(
    tmp_path: Path, monkeypatch
) -> None:
    """A still-running owned Swarm ends this parent turn before model drift."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _RunningSwarmTool()
    advisory = _RecordingAdvisoryTool("get_market_data")
    registry.register(swarm)
    registry.register(advisory)
    llm = _ResolveThenAdvisoryLLM("get_market_data")
    agent = _agent(tmp_path, llm, registry, max_iterations=5)
    monkeypatch.setattr(
        WorkflowObligationLedger,
        "prepare_run_swarm",
        lambda _self, arguments, _identity: arguments.update(
            {"preset_name": "quant_scalp_desk"}
        ) or None,
    )

    result = agent.run(user_message=STRICT_SWARM_REQUEST, session_id="session-running")

    assert swarm.calls == 1
    assert advisory.calls == 0
    assert llm.calls == 1
    assert result["status"] == "waiting"
    assert "still running" in result["content"]


def test_running_result_for_different_swarm_cannot_quiesce_owned_parent(
    tmp_path: Path,
) -> None:
    """A non-owned active result cannot hijack the parent's waiting handoff."""
    agent = _agent(tmp_path, _FinalTextLLM())
    ledger = WorkflowObligationLedger(
        run_dir=tmp_path / "run", user_message=STRICT_SWARM_REQUEST
    )
    ledger.mark_swarm_started()
    ledger.bind_swarm_run("swarm-owned")
    agent._workflow_obligation = ledger
    agent._trusted_owner_session_id = "session-owned"
    agent._active_swarm_run_id = "swarm-owned"
    agent._active_swarm_launch_id = ledger.obligation.launch_id
    agent._active_swarm_owner_session_id = "session-owned"
    agent._last_owned_swarm_tool_result = json.dumps(
        {
            "status": "running",
            "run_id": "swarm-other",
            "wait_budget_exhausted": True,
        }
    )

    assert agent._owned_swarm_active_handoff() is None


def test_owned_terminal_failed_swarm_stops_parent_before_advisory_tools(
    tmp_path: Path, monkeypatch
) -> None:
    """A direct owned terminal result ends the parent loop immediately."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _TerminalSwarmTool(
        {
            "status": "failed",
            "run_id": "swarm-terminal-failed",
            "error": (
                "backtest_artifact_contract_incomplete: "
                "missing=backtest.execution_provenance,backtest.config"
            ),
            "tasks": [
                {"id": "task-backtest", "status": "failed"},
                {"id": "task-risk", "status": "blocked"},
                {"id": "task-report", "status": "blocked"},
            ],
        }
    )
    advisory = _RecordingAdvisoryTool("get_market_data")
    registry.register(swarm)
    registry.register(advisory)
    agent = _agent(tmp_path, _ResolveThenAdvisoryLLM("get_market_data"), registry)
    monkeypatch.setattr(
        WorkflowObligationLedger,
        "prepare_run_swarm",
        lambda _self, arguments, _identity: arguments.update(
            {"preset_name": "quant_scalp_desk"}
        ) or None,
    )
    # This unit test exercises the direct tool-result handoff; canonical-store
    # reconciliation is covered separately and would pull optional MCP deps.
    agent._reconcile_owned_swarm_from_store = lambda: None

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert swarm.calls == 1
    assert advisory.calls == 0
    assert result["status"] == "failed"
    assert "backtest_artifact_contract_incomplete" in result["content"]
    calls = [
        event["tool"]
        for event in TraceWriter.read(tmp_path / "run")
        if event.get("type") == "tool_call"
    ]
    assert calls == ["search_symbol", "run_swarm"]


def test_owned_terminal_completed_swarm_returns_official_report_without_more_tools(
    tmp_path: Path, monkeypatch
) -> None:
    """Completed owned runs bypass another model turn and expose the report."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _TerminalSwarmTool(
        {
            "status": "completed",
            "run_id": "swarm-terminal-completed",
            "final_report": "Official limited-baseline conclusion.",
            "tasks": [],
        }
    )
    advisory = _RecordingAdvisoryTool("load_skill")
    registry.register(swarm)
    registry.register(advisory)
    agent = _agent(tmp_path, _ResolveThenAdvisoryLLM("load_skill"), registry)
    monkeypatch.setattr(
        WorkflowObligationLedger,
        "prepare_run_swarm",
        lambda _self, arguments, _identity: arguments.update(
            {"preset_name": "quant_scalp_desk"}
        ) or None,
    )
    agent._reconcile_owned_swarm_from_store = lambda: None

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert swarm.calls == 1
    assert advisory.calls == 0
    assert result["status"] == "success"
    assert result["content"] == "Official limited-baseline conclusion."


def test_owned_terminal_no_usable_bars_is_scoped_to_backtest_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed worker handoff cannot be paraphrased as global MT5 absence."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _TerminalSwarmTool(
        {
            "status": "failed",
            "run_id": "swarm-terminal-no-bars",
            "error": "backtest acquisition/handoff failed: MT5 returned no usable bars",
            "tasks": [{"id": "task-backtest", "status": "failed"}],
        }
    )
    registry.register(swarm)
    agent = _agent(tmp_path, _ResolveThenAdvisoryLLM("get_market_data"), registry)
    monkeypatch.setattr(
        WorkflowObligationLedger,
        "prepare_run_swarm",
        lambda _self, arguments, _identity: arguments.update(
            {"preset_name": "quant_scalp_desk"}
        ) or None,
    )
    agent._reconcile_owned_swarm_from_store = lambda: None

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert "backtest acquisition/handoff failed" in result["content"]
    assert "MT5 globally has no data" not in result["content"]


def test_unowned_terminal_result_does_not_stop_parent_loop(tmp_path: Path, monkeypatch) -> None:
    """A terminal response with the wrong run id cannot hijack this parent."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _TerminalSwarmTool(
        {
            "status": "failed",
            "run_id": "swarm-not-owned-by-parent",
            "terminal_reason": "unrelated terminal failure",
        },
        started_run_id="swarm-owned-by-parent",
    )
    advisory = _RecordingAdvisoryTool("get_market_data")
    registry.register(swarm)
    registry.register(advisory)
    agent = _agent(tmp_path, _ResolveThenAdvisoryLLM("get_market_data"), registry)
    monkeypatch.setattr(
        WorkflowObligationLedger,
        "prepare_run_swarm",
        lambda _self, arguments, _identity: arguments.update(
            {"preset_name": "quant_scalp_desk"}
        ) or None,
    )
    agent._reconcile_owned_swarm_from_store = lambda: None

    agent.run(user_message=STRICT_SWARM_REQUEST)

    assert swarm.calls == 1
    calls = [
        event["tool"]
        for event in TraceWriter.read(tmp_path / "run")
        if event.get("type") == "tool_call"
    ]
    assert "get_market_data" in calls
    assert not any(
        event.get("type") == "owned_swarm_terminal_handoff"
        for event in TraceWriter.read(tmp_path / "run")
    )


def test_verified_strict_swarm_dispatches_before_local_workaround_skill(tmp_path: Path) -> None:
    """A mid-run advisory skill cannot delay the server-owned dispatch."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _RecordingSwarmTool()
    advisory = _RecordingAdvisoryTool("load_skill")
    registry.register(swarm)
    registry.register(advisory)
    agent = _agent(tmp_path, _ResolveThenAdvisoryLLM("load_skill"), registry, max_iterations=5)

    agent.run(user_message=STRICT_SWARM_REQUEST)

    assert len(swarm.calls) == 1
    assert advisory.calls == 0


def test_strict_swarm_suppresses_workaround_skill_while_identity_is_pending(tmp_path: Path) -> None:
    """The resolver precondition cannot be bundled with advisory skill drift."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _RecordingSwarmTool()
    advisory = _RecordingAdvisoryTool("load_skill")
    registry.register(swarm)
    registry.register(advisory)
    agent = _agent(tmp_path, _AdvisoryAndResolveLLM("load_skill"), registry, max_iterations=5)

    agent.run(user_message=STRICT_SWARM_REQUEST)

    assert advisory.calls == 0
    assert len(swarm.calls) == 1


def test_verified_strict_swarm_without_dispatch_tool_fails_fast(tmp_path: Path) -> None:
    """Removing run_swarm must create a structured blocker, not advisory drift."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    advisory = _RecordingAdvisoryTool("get_market_data")
    registry.register(advisory)
    llm = _ResolveThenAdvisoryLLM("get_market_data")
    agent = _agent(tmp_path, llm, registry, max_iterations=12)

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert result["status"] == "failed"
    assert result["reason"] == "run_swarm_unavailable"
    assert advisory.calls == 0
    assert llm.calls == 1


def test_unavailable_required_attachment_blocks_before_strict_swarm_dispatch(tmp_path: Path) -> None:
    """A known-unavailable uploaded strategy document is a hard blocker."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    registry.register(_UnavailableDocumentTool())
    swarm = _RecordingSwarmTool()
    registry.register(swarm)
    llm = _ResolveWithUnavailableAttachmentLLM()
    agent = _agent(tmp_path, llm, registry, max_iterations=12)

    result = agent.run(user_message=STRICT_SWARM_REQUEST_WITH_ATTACHMENT)

    assert result["status"] == "failed"
    assert result["reason"] == "required_attachment_unavailable"
    assert swarm.calls == []
    assert llm.calls == 1
