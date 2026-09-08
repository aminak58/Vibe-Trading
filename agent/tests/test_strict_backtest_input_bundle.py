"""Strict server-owned backtest worker input-bundle contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from src.agent.tools import BaseTool, ToolRegistry
from src.execution_identity import (
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionResolution,
    FallbackPolicy,
    SourceMode,
    SyntheticDataPolicy,
)
from src.swarm.strict_backtest_bundle import (
    StrategySource,
    build_strict_backtest_input_bundle,
    materialize_strategy_source,
    strategy_source_from_user_vars,
    strategy_source_payload,
    validate_strict_backtest_package,
)
from src.tools.swarm_tool import _attach_strict_strategy_source
import src.swarm.worker as worker_mod
from src.providers.chat import LLMResponse, ToolCallRequest
from src.swarm.models import SwarmAgentSpec, SwarmTask
from src.swarm.worker import _strict_backtest_bundle_for_worker, agent_artifact_dir, run_worker


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-backtest-bundle-test",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
            synthetic=SyntheticDataPolicy.FORBID,
        ),
        requests=(ExecutionRequest(request_id="gold", symbol="XAUUSD", source="mt5"),),
        resolutions=(
            ExecutionResolution(
                request_id="gold",
                resolved_symbol="XAUUSD_o",
                source="mt5",
                resolver_evidence_ref="tool:search_symbol:test",
            ),
        ),
    )


def _source(*, available: bool = True) -> StrategySource:
    return StrategySource(
        status="available" if available else "unavailable",
        evidence_ref="document:strategy.pdf" if available else None,
        content_hash="a" * 64 if available else None,
        materialized_path="strategy_source.txt" if available else None,
    )


@pytest.fixture
def bundle():
    return build_strict_backtest_input_bundle(
        _identity(),
        window_authority={"source": "unknown", "user_explicit": False},
        strategy_source=_source(),
    )


def _write_config(tmp_path: Path, config: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _valid_config() -> dict:
    return {
        "codes": ["XAUUSD_o"],
        "source": "mt5",
        "start_date": "2026-08-20",
        "end_date": "2026-09-07",
        "interval": "5m",
    }


def test_verified_mt5_identity_builds_canonical_bundle(bundle) -> None:
    assert bundle.requested_symbol == "XAUUSD"
    assert bundle.resolved_symbol == "XAUUSD_o"
    assert bundle.codes == ("XAUUSD_o",)
    assert bundle.source == "mt5"
    assert bundle.source_mode == "strict"
    assert bundle.fallback == "deny"
    assert bundle.synthetic == "forbid"
    assert bundle.expected_config_path == "config.json"
    assert bundle.expected_strategy_path == "code/signal_engine.py"


@pytest.mark.parametrize("field", ["code", "symbol", "resolved_symbol"])
def test_non_contract_symbol_fields_do_not_satisfy_codes(tmp_path: Path, bundle, field: str) -> None:
    _write_config(
        tmp_path,
        {"source": "mt5", field: "XAUUSD_o", "start_date": "2026-08-20", "end_date": "2026-09-07"},
    )

    result = validate_strict_backtest_package(tmp_path, bundle)

    assert result is not None
    assert result["error_code"] == "invalid_backtest_config_identity"


def test_wrong_strategy_path_fails_before_backtest(tmp_path: Path, bundle) -> None:
    _write_config(tmp_path, _valid_config())
    (tmp_path / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")

    result = validate_strict_backtest_package(tmp_path, bundle)

    assert result is not None
    assert result["error_code"] == "invalid_backtest_package_path"


def test_required_strategy_source_unavailable_fails_before_package_checks(tmp_path: Path) -> None:
    unavailable_bundle = build_strict_backtest_input_bundle(
        _identity(), window_authority={}, strategy_source=_source(available=False)
    )

    result = validate_strict_backtest_package(tmp_path, unavailable_bundle)

    assert result is not None
    assert result["error_code"] == "required_strategy_source_unavailable"


@pytest.mark.parametrize(
    "dates",
    [
        {"start_date": "", "end_date": "2026-09-07"},
        {"start_date": "2026-08-20", "end_date": "2026-2026-09-07"},
        {"start_date": "2026-09-07", "end_date": "2026-08-20"},
    ],
)
def test_invalid_window_is_blocked_before_acquisition(tmp_path: Path, bundle, dates: dict[str, str]) -> None:
    config = _valid_config() | dates
    _write_config(tmp_path, config)
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")

    result = validate_strict_backtest_package(tmp_path, bundle)

    assert result is not None
    assert result["error_code"] == "invalid_window_config"


def test_valid_strict_package_passes_preflight(tmp_path: Path, bundle) -> None:
    _write_config(tmp_path, _valid_config())
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")

    assert validate_strict_backtest_package(tmp_path, bundle) is None


def test_server_owned_strategy_text_is_materialized_with_a_hash(tmp_path: Path) -> None:
    source = materialize_strategy_source(
        tmp_path,
        text="Authoritative VWAP strategy rules.",
        evidence_ref="uploads/strategy.pdf",
    )

    assert source.status == "available"
    assert source.content_hash is not None
    assert len(source.content_hash) == 64
    assert source.materialized_path == "strategy_source.txt"
    assert (tmp_path / source.materialized_path).read_text(encoding="utf-8") == "Authoritative VWAP strategy rules."


def test_raw_upload_handle_in_goal_does_not_establish_strategy_authority(tmp_path: Path) -> None:
    source = strategy_source_from_user_vars(
        {"goal": "[Uploaded file: strategy.pdf, path: uploads/strategy.pdf]"}, tmp_path
    )

    assert source.status == "unavailable"


def test_server_owned_strategy_payload_materializes_for_worker(tmp_path: Path) -> None:
    payload = strategy_source_payload("Authoritative VWAP rules.", "uploads/strategy.pdf")
    source = strategy_source_from_user_vars({"__strict_strategy_source_v1": payload}, tmp_path)

    assert source.status == "available"
    assert source.evidence_ref == "uploads/strategy.pdf"
    assert (tmp_path / "strategy_source.txt").read_text(encoding="utf-8") == "Authoritative VWAP rules."


def test_swarm_tool_transports_only_successful_extracted_document(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.tools.doc_reader_tool.read_document",
        lambda _path: json.dumps({"status": "ok", "text": "VWAP source", "file": "uploads/strategy.pdf"}),
    )
    variables = {"goal": "[Uploaded file: strategy.pdf, path: uploads/strategy.pdf]"}

    _attach_strict_strategy_source(variables)

    assert "__strict_strategy_source_v1" in variables


def test_worker_uses_only_transport_payload_for_strict_bundle(tmp_path: Path) -> None:
    payload = strategy_source_payload("VWAP source", "uploads/strategy.pdf")

    bundle = _strict_backtest_bundle_for_worker(
        _identity(), {"__strict_strategy_source_v1": payload}, tmp_path, window_authority={}
    )

    assert bundle is not None
    assert bundle.strategy_source.status == "available"


class _BacktestSpyTool(BaseTool):
    """Delegated backtest stand-in used to prove the worker boundary.

    The test intentionally spies on ``execute`` rather than on the pure
    preflight helper: a structural failure must not reach the delegated tool
    at all.
    """

    name = "backtest"
    description = "Mock backtest tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps({"status": "ok", "mock": "backtest-ran"})


class _AcquisitionFailureSpyTool(_BacktestSpyTool):
    """A delegated strict acquisition failure with no package mutation."""

    def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "status": "error",
                "error_code": "mt5_copy_rates_empty",
                "stage": "copy_rates_range",
                "diagnostic": {"raw_row_count": 0},
            }
        )


class _ScriptedWorkerLLM:
    """Small ChatLLM stand-in that records messages across worker turns."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.received_messages: list[list[dict]] = []

    def __call__(self, *args, **kwargs) -> "_ScriptedWorkerLLM":
        return self

    def close(self) -> None:
        pass

    def stream_chat(self, messages, tools=None, on_text_chunk=None, timeout=None, **kwargs) -> LLMResponse:
        self.received_messages.append(list(messages))
        return self._responses.pop(0)


class _PackageMutatingWorkerLLM(_ScriptedWorkerLLM):
    """Simulate a worker repairing its package after a tool-visible failure."""

    def __init__(
        self, responses: list[LLMResponse], *, before_response: dict[int, Callable[[], None]]
    ) -> None:
        super().__init__(responses)
        self._before_response = before_response

    def stream_chat(self, messages, tools=None, on_text_chunk=None, timeout=None, **kwargs) -> LLMResponse:
        response_number = len(self.received_messages) + 1
        callback = self._before_response.get(response_number)
        if callback is not None:
            callback()
        return super().stream_chat(
            messages,
            tools=tools,
            on_text_chunk=on_text_chunk,
            timeout=timeout,
            **kwargs,
        )


def _strict_backtest_agent() -> SwarmAgentSpec:
    return SwarmAgentSpec(
        id="backtester",
        role="Strict strategy backtester",
        system_prompt="Use the server-owned strict backtest bundle.",
        tools=["backtest"],
        skills=[],
        max_iterations=3,
        timeout_seconds=60,
    )


def _strict_backtest_task() -> SwarmTask:
    return SwarmTask(
        id="task-backtest",
        agent_id="backtester",
        prompt_template="{goal}",
    )


def _worker_user_vars() -> dict[str, str]:
    return {
        "goal": "Evaluate the attached strategy under the strict contract.",
        "__strict_strategy_source_v1": strategy_source_payload(
            "Authoritative VWAP strategy rules.", "artifact:strategy-source"
        ),
    }


def _run_strict_worker(
    monkeypatch,
    tmp_path: Path,
    llm: _ScriptedWorkerLLM,
    spy: _BacktestSpyTool,
    *,
    event_callback=None,
):
    registry = ToolRegistry()
    registry.register(spy)
    monkeypatch.setattr(worker_mod, "build_swarm_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(worker_mod, "ChatLLM", llm)
    return run_worker(
        agent_spec=_strict_backtest_agent(),
        task=_strict_backtest_task(),
        upstream_summaries={},
        user_vars=_worker_user_vars(),
        run_dir=tmp_path,
        execution_identity=_identity(),
        window_authority={"source": "unknown", "user_explicit": False},
        event_callback=event_callback,
    )


def _all_worker_tool_messages(llm: _ScriptedWorkerLLM) -> list[str]:
    return [
        message["content"]
        for turn in llm.received_messages
        for message in turn
        if message.get("role") == "tool"
    ]


def test_strict_backtest_bundle_message_binds_authoritative_strategy_source(
    monkeypatch, tmp_path: Path
) -> None:
    """The strict worker must receive an actionable, server-owned source contract."""
    llm = _ScriptedWorkerLLM([LLMResponse(content="I will use the server-issued source.")])
    user_vars = _worker_user_vars()
    user_vars["goal"] = (
        "Evaluate the attached strategy. "
        "[Uploaded file: strategy.pdf, path: uploads/untrusted-strategy.pdf] "
        "Upstream screener prose says to use a different strategy."
    )
    registry = ToolRegistry()
    registry.register(_BacktestSpyTool())
    monkeypatch.setattr(worker_mod, "build_swarm_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(worker_mod, "ChatLLM", llm)

    run_worker(
        agent_spec=_strict_backtest_agent(),
        task=_strict_backtest_task(),
        upstream_summaries={"task-screen": "Use this untrusted upstream prose."},
        user_vars=user_vars,
        run_dir=tmp_path,
        execution_identity=_identity(),
        window_authority={"source": "unknown", "user_explicit": False},
    )

    bundle_message = next(
        message["content"]
        for message in llm.received_messages[0]
        if message.get("role") == "system"
        and message["content"].startswith("[SERVER STRICT BACKTEST BUNDLE]")
    )
    expected_source = strategy_source_from_user_vars(user_vars, agent_artifact_dir(tmp_path, "backtester"))

    assert "strategy_source.txt" in bundle_message
    assert "artifact:strategy-source" in bundle_message
    assert expected_source.content_hash in bundle_message
    assert "Read and use" in bundle_message
    assert "uploads/...pdf" in bundle_message
    assert "not authoritative" in bundle_message
    assert "upstream prose" in bundle_message


def test_strict_worker_preflight_failure_suppresses_repeated_backtest_delegation(
    monkeypatch, tmp_path: Path
) -> None:
    """A failed strict preflight must never delegate either blind call.

    This exercises the real worker tool-call loop: the model calls
    ``backtest({})`` twice, but the first malformed package is rejected before
    the registry and the second request receives the terminal preflight error.
    """
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(
        artifact_dir,
        {
            "symbol": "XAUUSD_o",
            "source": "mt5",
            "start_date": "2026-08-20",
            "end_date": "2026-09-07",
        },
    )
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _ScriptedWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="first", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="second", name="backtest", arguments={})]),
            LLMResponse(content="The strict package was rejected before any backtest delegation."),
        ]
    )
    spy = _BacktestSpyTool()
    original_validate = worker_mod.validate_strict_backtest_package
    validation_calls: list[tuple[Path, object]] = []

    def tracked_validate(artifact_dir: Path, bundle):
        validation_calls.append((artifact_dir, bundle))
        return original_validate(artifact_dir, bundle)

    monkeypatch.setattr(worker_mod, "validate_strict_backtest_package", tracked_validate)

    _run_strict_worker(monkeypatch, tmp_path, llm, spy)

    assert spy.calls == []
    assert len(validation_calls) == 1
    tool_messages = "\n".join(_all_worker_tool_messages(llm))
    assert "invalid_backtest_config_identity" in tool_messages
    assert "strict_backtest_preflight_already_failed" in tool_messages


def test_strict_worker_revalidates_a_repaired_package_before_single_delegation(
    monkeypatch, tmp_path: Path
) -> None:
    """A changed strict package must escape only its old preflight failure.

    The production bug was a worker-wide ``strict_preflight_failed`` flag:
    after the model wrote the required ``codes`` field, the repaired package
    was still suppressed without another validation or delegated backtest.
    """
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(
        artifact_dir,
        {
            "symbol": "XAUUSD_o",
            "source": "mt5",
            "start_date": "2026-08-20",
            "end_date": "2026-09-07",
        },
    )
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )

    def repair_config() -> None:
        _write_config(artifact_dir, _valid_config())

    llm = _PackageMutatingWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="invalid", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="repaired", name="backtest", arguments={})]),
            LLMResponse(content="The repaired strict package reached the delegated backtest."),
        ],
        before_response={2: repair_config},
    )
    spy = _BacktestSpyTool()
    original_validate = worker_mod.validate_strict_backtest_package
    validation_calls: list[Path] = []

    def tracked_validate(path: Path, bundle):
        validation_calls.append(path)
        return original_validate(path, bundle)

    monkeypatch.setattr(worker_mod, "validate_strict_backtest_package", tracked_validate)

    _run_strict_worker(monkeypatch, tmp_path, llm, spy)

    assert len(validation_calls) == 2
    assert len(spy.calls) == 1
    tool_messages = "\n".join(_all_worker_tool_messages(llm))
    assert "invalid_backtest_config_identity" in tool_messages
    assert "strict_backtest_preflight_already_failed" not in tool_messages


def test_strict_worker_revalidates_repaired_dates_without_expanding_date_grammar(
    monkeypatch, tmp_path: Path
) -> None:
    """Repairing an invalid ISO timestamp to date-only form may reach backtest."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(
        artifact_dir,
        _valid_config() | {"start_date": "2026-08-20T00:00:00Z"},
    )
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )

    llm = _PackageMutatingWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="invalid", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="repaired", name="backtest", arguments={})]),
            LLMResponse(content="The repaired date-only package reached the delegated backtest."),
        ],
        before_response={2: lambda: _write_config(artifact_dir, _valid_config())},
    )
    spy = _BacktestSpyTool()

    _run_strict_worker(monkeypatch, tmp_path, llm, spy)

    assert len(spy.calls) == 1
    tool_messages = "\n".join(_all_worker_tool_messages(llm))
    assert "invalid_window_config" in tool_messages


def test_strict_worker_keeps_same_preflight_failure_when_only_untracked_file_changes(
    monkeypatch, tmp_path: Path
) -> None:
    """Changing unrelated worker output must not evade identical-package suppression."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(
        artifact_dir,
        {
            "symbol": "XAUUSD_o",
            "source": "mt5",
            "start_date": "2026-08-20",
            "end_date": "2026-09-07",
        },
    )
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _PackageMutatingWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="first", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="second", name="backtest", arguments={})]),
            LLMResponse(content="The same invalid package remained blocked."),
        ],
        before_response={2: lambda: (artifact_dir / "notes.txt").write_text("unrelated", encoding="utf-8")},
    )
    spy = _BacktestSpyTool()
    original_validate = worker_mod.validate_strict_backtest_package
    validation_calls: list[Path] = []

    def tracked_validate(path: Path, bundle):
        validation_calls.append(path)
        return original_validate(path, bundle)

    monkeypatch.setattr(worker_mod, "validate_strict_backtest_package", tracked_validate)

    _run_strict_worker(monkeypatch, tmp_path, llm, spy)

    assert spy.calls == []
    assert len(validation_calls) == 1
    tool_messages = "\n".join(_all_worker_tool_messages(llm))
    assert "strict_backtest_preflight_already_failed" in tool_messages


def test_strict_worker_records_failed_and_revalidated_package_fingerprints(
    monkeypatch, tmp_path: Path
) -> None:
    """Audit events must distinguish an old invalid package from its repair."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(
        artifact_dir,
        {
            "symbol": "XAUUSD_o",
            "source": "mt5",
            "start_date": "2026-08-20",
            "end_date": "2026-09-07",
        },
    )
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _PackageMutatingWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="invalid", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="repaired", name="backtest", arguments={})]),
            LLMResponse(content="The repaired package was delegated once."),
        ],
        before_response={2: lambda: _write_config(artifact_dir, _valid_config())},
    )
    events = []

    _run_strict_worker(
        monkeypatch,
        tmp_path,
        llm,
        _BacktestSpyTool(),
        event_callback=events.append,
    )

    results = [
        event.data
        for event in events
        if event.type == "tool_result" and event.data.get("tool") == "backtest"
    ]
    assert [result["preflight_status"] for result in results] == ["failed", "revalidated"]
    assert results[0]["preflight_fingerprint"] != results[1]["preflight_fingerprint"]


def test_strict_worker_emits_tool_result_for_every_preflight_block(monkeypatch, tmp_path: Path) -> None:
    """Both blocked backtest calls must leave auditable error events."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(
        artifact_dir,
        {
            "resolved_symbol": "XAUUSD_o",
            "source": "mt5",
            "start_date": "2026-08-20",
            "end_date": "2026-09-07",
        },
    )
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _ScriptedWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="first", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="second", name="backtest", arguments={})]),
            LLMResponse(content="No delegated strict backtest was run."),
        ]
    )
    events = []

    _run_strict_worker(
        monkeypatch,
        tmp_path,
        llm,
        _BacktestSpyTool(),
        event_callback=events.append,
    )

    results = [
        event.data
        for event in events
        if event.type == "tool_result" and event.data.get("tool") == "backtest"
    ]
    assert [result["status"] for result in results] == ["error", "error"]
    assert [result["error_code"] for result in results] == [
        "invalid_backtest_config_identity",
        "strict_backtest_preflight_already_failed",
    ]
    assert results[1]["original_error_code"] == "invalid_backtest_config_identity"
    assert results[1]["preflight_fingerprint"] == results[0]["preflight_fingerprint"]
    assert results[1]["original_failure_fingerprint"] == results[0]["preflight_fingerprint"]
    assert results[1]["suppression_code"] == "strict_backtest_preflight_already_failed"


def test_strict_worker_valid_package_delegates_backtest_once(monkeypatch, tmp_path: Path) -> None:
    """A valid strict package remains able to reach the delegated tool once."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(artifact_dir, _valid_config())
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _ScriptedWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="valid", name="backtest", arguments={})]),
            LLMResponse(content="The mocked strict backtest completed successfully."),
        ]
    )
    spy = _BacktestSpyTool()

    _run_strict_worker(monkeypatch, tmp_path, llm, spy)

    assert len(spy.calls) == 1
    assert spy.calls[0]["run_dir"] == str(artifact_dir)
    assert spy.calls[0]["__execution_identity"] == _identity()


def test_strict_worker_suppresses_second_identical_acquisition_failure(monkeypatch, tmp_path: Path) -> None:
    """A terminal strict acquisition failure must delegate only its first call."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(artifact_dir, _valid_config())
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _ScriptedWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="first", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="second", name="backtest", arguments={})]),
            LLMResponse(content="The strict acquisition failure was reported once."),
        ]
    )
    spy = _AcquisitionFailureSpyTool()

    _run_strict_worker(monkeypatch, tmp_path, llm, spy)

    assert len(spy.calls) == 1
    tool_messages = "\n".join(_all_worker_tool_messages(llm))
    assert "strict_backtest_acquisition_failure_already_seen" in tool_messages
    assert "mt5_copy_rates_empty" in tool_messages


def test_strict_worker_emits_structured_events_for_acquisition_suppression(
    monkeypatch, tmp_path: Path
) -> None:
    """Both the original acquisition failure and suppression must be auditable."""
    artifact_dir = agent_artifact_dir(tmp_path, "backtester")
    artifact_dir.mkdir(parents=True)
    _write_config(artifact_dir, _valid_config())
    (artifact_dir / "code").mkdir()
    (artifact_dir / "code" / "signal_engine.py").write_text(
        "class SignalEngine: pass\n", encoding="utf-8"
    )
    llm = _ScriptedWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="first", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="second", name="backtest", arguments={})]),
            LLMResponse(content="The strict acquisition failure was reported once."),
        ]
    )
    events = []

    _run_strict_worker(
        monkeypatch,
        tmp_path,
        llm,
        _AcquisitionFailureSpyTool(),
        event_callback=events.append,
    )

    results = [
        event.data
        for event in events
        if event.type == "tool_result" and event.data.get("tool") == "backtest"
    ]
    assert [result["error_code"] for result in results] == [
        "mt5_copy_rates_empty",
        "strict_backtest_acquisition_failure_already_seen",
    ]
    assert results[1]["original_error_code"] == "mt5_copy_rates_empty"


def test_non_strict_worker_does_not_suppress_repeated_acquisition_failures(
    monkeypatch, tmp_path: Path
) -> None:
    """The strict acquisition guard must not change generic worker behavior."""
    registry = ToolRegistry()
    spy = _AcquisitionFailureSpyTool()
    registry.register(spy)
    llm = _ScriptedWorkerLLM(
        [
            LLMResponse(tool_calls=[ToolCallRequest(id="first", name="backtest", arguments={})]),
            LLMResponse(tool_calls=[ToolCallRequest(id="second", name="backtest", arguments={})]),
            LLMResponse(content="Generic worker reported both tool failures."),
        ]
    )
    monkeypatch.setattr(worker_mod, "build_swarm_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr(worker_mod, "ChatLLM", llm)

    run_worker(
        agent_spec=_strict_backtest_agent(),
        task=_strict_backtest_task(),
        upstream_summaries={},
        user_vars={"goal": "Generic backtest."},
        run_dir=tmp_path,
        execution_identity=None,
    )

    assert len(spy.calls) == 2
