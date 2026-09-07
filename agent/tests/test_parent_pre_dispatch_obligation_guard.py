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


def test_verified_strict_swarm_dispatches_before_advisory_market_data(tmp_path: Path) -> None:
    """A model cannot spend its post-identity turn on a coverage probe."""
    registry = ToolRegistry()
    registry.register(_StrictMt5ResolverTool())
    swarm = _RecordingSwarmTool()
    advisory = _RecordingAdvisoryTool("get_market_data")
    registry.register(swarm)
    registry.register(advisory)
    llm = _ResolveThenAdvisoryLLM("get_market_data")
    agent = _agent(tmp_path, llm, registry, max_iterations=5)

    result = agent.run(user_message=STRICT_SWARM_REQUEST)

    assert len(swarm.calls) == 1
    calls = [
        event["tool"]
        for event in TraceWriter.read(tmp_path / "run")
        if event.get("type") == "tool_call"
    ]
    assert calls.index("run_swarm") < calls.index("get_market_data")
    assert swarm.calls[0]["prompt"] == STRICT_SWARM_REQUEST
    assert swarm.calls[0]["__execution_identity"].resolutions[0].resolved_symbol == "XAUUSD_o"
    assert result["status"] == "failed"


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
