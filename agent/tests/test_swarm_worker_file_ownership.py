"""Worker-owned file writes must stay inside one agent artifact directory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.providers.chat import LLMResponse, ToolCallRequest
from src.swarm.models import SwarmAgentSpec, SwarmEvent, SwarmTask
import src.swarm.worker as worker_mod
from src.swarm.worker import run_worker
from src.tools.backtest_tool import BacktestTool
from src.tools.edit_file_tool import EditFileTool
from src.tools.write_file_tool import WriteFileTool


FINAL_TEXT = "# Worker report\n\nThe scripted worker completed its file operations."


class _ScriptedLLM:
    """Return scripted tool calls, then a normal final worker response."""

    def __init__(self, calls: list[ToolCallRequest]) -> None:
        self._responses = [LLMResponse(tool_calls=calls), LLMResponse(content=FINAL_TEXT)]

    def __call__(self, *args, **kwargs) -> "_ScriptedLLM":
        return self

    def close(self) -> None:
        return None

    def stream_chat(self, messages, tools=None, on_text_chunk=None, timeout=None):
        return self._responses.pop(0)


class _RealFileToolRegistry:
    """Spy dispatches while retaining the real write/edit tool behavior."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []
        self.results: list[tuple[str, str]] = []
        self._tools = {"write_file": WriteFileTool(), "edit_file": EditFileTool()}

    def get_definitions(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in self._tools
        ]

    def get(self, name: str):
        return self._tools.get(name)

    def execute(self, name: str, args: dict) -> str:
        self.executed.append((name, dict(args)))
        result = self._tools[name].execute(**args)
        self.results.append((name, result))
        return result


class _RealBacktestRegistry(_RealFileToolRegistry):
    """Use real backtest preflight; invalid config stops before acquisition."""

    def __init__(self) -> None:
        super().__init__()
        self._tools["backtest"] = BacktestTool()

def _call(name: str, **arguments: object) -> ToolCallRequest:
    return ToolCallRequest(id=f"call-{name}-{len(arguments)}", name=name, arguments=arguments)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[ToolCallRequest],
    registry: _RealFileToolRegistry,
) -> tuple[object, list[SwarmEvent]]:
    events: list[SwarmEvent] = []
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_RUN_ROOTS", str(tmp_path))
    agent = SwarmAgentSpec(
        id="analyst",
        role="Analyst",
        system_prompt="Write the requested artifacts.",
        tools=["write_file", "edit_file", "backtest"],
        skills=[],
        max_iterations=3,
        timeout_seconds=60,
    )
    task = SwarmTask(id="task-1", agent_id="analyst", prompt_template="Work.")
    with (
        patch.object(worker_mod, "build_swarm_registry", lambda *args, **kwargs: registry),
        patch.object(worker_mod, "ChatLLM", _ScriptedLLM(calls)),
    ):
        result = run_worker(
            agent_spec=agent,
            task=task,
            upstream_summaries={},
            user_vars={},
            run_dir=tmp_path / "swarm-run",
            event_callback=events.append,
        )
    return result, events


def _artifact_dir(tmp_path: Path) -> Path:
    return tmp_path / "swarm-run" / "artifacts" / "analyst"


def test_worker_writes_and_edits_only_its_canonical_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    """Relative and absolute inner paths work, and tools receive canonical paths."""
    artifact_dir = _artifact_dir(tmp_path)
    absolute_target = artifact_dir / "nested" / "absolute.md"
    registry = _RealFileToolRegistry()

    result, events = _run(
        monkeypatch,
        tmp_path,
        [
            _call("write_file", path="relative.md", content="before", run_dir="spoofed"),
            _call("write_file", path=str(absolute_target), content="old"),
            _call("edit_file", path="relative.md", old_text="before", new_text="after"),
            _call("edit_file", path=str(absolute_target), old_text="old", new_text="new"),
            _call("write_file", path="report.md", content=FINAL_TEXT),
        ],
        registry,
    )

    assert result.status == "completed"
    assert (artifact_dir / "relative.md").read_text(encoding="utf-8") == "after"
    assert absolute_target.read_text(encoding="utf-8") == "new"
    assert all(Path(args["run_dir"]) == artifact_dir for _, args in registry.executed)
    assert all(Path(args["path"]).is_absolute() for _, args in registry.executed)
    assert all(Path(args["path"]).is_relative_to(artifact_dir) for _, args in registry.executed)
    assert [event.data["status"] for event in events if event.type == "tool_result"] == ["ok"] * 5


@pytest.mark.parametrize("alias", ["file_path", "filepath", "filename", "file"])
def test_worker_preserves_write_file_aliases_inside_artifact_dir(monkeypatch, tmp_path: Path, alias: str) -> None:
    """Every write-file alias remains usable after worker path canonicalization."""
    registry = _RealFileToolRegistry()
    result, _ = _run(
        monkeypatch,
        tmp_path,
        [_call("write_file", **{alias: f"{alias}.md", "content": alias}), _call("write_file", path="report.md", content=FINAL_TEXT)],
        registry,
    )

    assert result.status == "completed"
    assert (_artifact_dir(tmp_path) / f"{alias}.md").read_text(encoding="utf-8") == alias


def test_worker_preserves_path_alias_precedence(monkeypatch, tmp_path: Path) -> None:
    """A supplied path keeps precedence over lower-priority write-file aliases."""
    registry = _RealFileToolRegistry()
    _run(
        monkeypatch,
        tmp_path,
        [
            _call("write_file", path="path-wins.md", file="must-not-be-used.md", content="ok"),
            _call("write_file", path="report.md", content=FINAL_TEXT),
        ],
        registry,
    )

    assert (_artifact_dir(tmp_path) / "path-wins.md").read_text(encoding="utf-8") == "ok"
    assert not (_artifact_dir(tmp_path) / "must-not-be-used.md").exists()


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda artifact, parent, sibling, global_root: "../parent.md",
        lambda artifact, parent, sibling, global_root: str(parent / "parent.md"),
        lambda artifact, parent, sibling, global_root: str(sibling / "sibling.md"),
        lambda artifact, parent, sibling, global_root: str(global_root / "global.md"),
    ],
    ids=["traversal", "parent", "sibling", "global_allowed"],
)
def test_worker_rejects_file_escapes_before_registry_dispatch(monkeypatch, tmp_path: Path, path_factory) -> None:
    """Worker writes may not use resolver fallback roots or an enclosing run directory."""
    artifact_dir = _artifact_dir(tmp_path)
    parent = artifact_dir.parent
    sibling = parent / "other-agent"
    global_root = tmp_path / "globally-allowed"
    sibling.mkdir(parents=True)
    global_root.mkdir()
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_WRITE_ROOTS", str(global_root))
    attempted = Path(path_factory(artifact_dir, parent, sibling, global_root))
    if not attempted.is_absolute():
        attempted = artifact_dir / attempted
    attempted.parent.mkdir(parents=True, exist_ok=True)
    attempted.write_text("protected", encoding="utf-8")
    registry = _RealFileToolRegistry()

    result, events = _run(
        monkeypatch,
        tmp_path,
        [
            _call("write_file", path=path_factory(artifact_dir, parent, sibling, global_root), content="blocked", run_dir=str(global_root)),
            _call("edit_file", path=path_factory(artifact_dir, parent, sibling, global_root), old_text="x", new_text="y"),
        ],
        registry,
    )

    assert registry.executed == []
    assert attempted.read_text(encoding="utf-8") == "protected"
    assert result.status == "incomplete"
    tool_results = [event for event in events if event.type == "tool_result"]
    assert len(tool_results) == 2
    assert all(event.data["status"] == "error" for event in tool_results)


@pytest.mark.parametrize("alias", ["file_path", "filepath", "filename", "file"])
def test_worker_rejects_write_file_alias_escapes_before_dispatch(monkeypatch, tmp_path: Path, alias: str) -> None:
    """No lower-priority write-file alias can bypass the artifact boundary."""
    outside = tmp_path / "globally-allowed"
    outside.mkdir()
    protected = outside / "protected.md"
    protected.write_text("protected", encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_WRITE_ROOTS", str(outside))
    registry = _RealFileToolRegistry()

    _, events = _run(
        monkeypatch,
        tmp_path,
        [_call("write_file", **{alias: str(protected), "content": "blocked"})],
        registry,
    )

    assert registry.executed == []
    assert protected.read_text(encoding="utf-8") == "protected"
    assert [event.data["status"] for event in events if event.type == "tool_result"] == ["error"]


def test_worker_rejection_does_not_create_escape_parent_directories(monkeypatch, tmp_path: Path) -> None:
    """Rejected traversal cannot create a new directory beside the worker artifacts."""
    registry = _RealFileToolRegistry()
    escaped_parent = _artifact_dir(tmp_path).parent / "new-parent"

    _run(
        monkeypatch,
        tmp_path,
        [_call("write_file", path="../new-parent/blocked.md", content="blocked")],
        registry,
    )

    assert registry.executed == []
    assert not escaped_parent.exists()


@pytest.mark.parametrize("bad_path", [None, 7, "", "\\\\server\\share\\file.md"])
def test_worker_rejects_malformed_file_paths_without_dispatch(monkeypatch, tmp_path: Path, bad_path: object) -> None:
    """Malformed file-path arguments become recoverable tool errors, never exceptions."""
    registry = _RealFileToolRegistry()

    _, events = _run(
        monkeypatch,
        tmp_path,
        [_call("write_file", path=bad_path, content="blocked")],
        registry,
    )

    assert registry.executed == []
    tool_results = [event for event in events if event.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].data["status"] == "error"


def test_worker_rejects_symlink_escape_before_real_file_tool(monkeypatch, tmp_path: Path) -> None:
    """A symlink inside artifacts that resolves outward cannot redirect a write."""
    artifact_dir = _artifact_dir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_dir.mkdir(parents=True)
    link = artifact_dir / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable on this platform: {exc}")
    registry = _RealFileToolRegistry()

    _, events = _run(
        monkeypatch,
        tmp_path,
        [_call("write_file", path="escape/outside.md", content="blocked")],
        registry,
    )

    assert registry.executed == []
    assert not (outside / "outside.md").exists()
    assert [event.data["status"] for event in events if event.type == "tool_result"] == ["error"]


def test_worker_real_backtest_preflight_reads_config_from_its_artifacts(monkeypatch, tmp_path: Path) -> None:
    """Real backtest preflight reads the config written under its injected artifact dir."""
    registry = _RealBacktestRegistry()
    config = {"source": "not-a-real-source", "codes": ["AAPL"]}

    result, _ = _run(
        monkeypatch,
        tmp_path,
        [
            _call("write_file", path="config.json", content=json.dumps(config)),
            _call("backtest", run_dir=str(tmp_path / "spoofed-parent")),
            _call("write_file", path="report.md", content=FINAL_TEXT),
        ],
        registry,
    )

    assert result.status == "completed"
    backtest_args = next(args for name, args in registry.executed if name == "backtest")
    assert Path(backtest_args["run_dir"]) == _artifact_dir(tmp_path)
    backtest_result = json.loads(next(value for name, value in registry.results if name == "backtest"))
    assert "source must be one of" in backtest_result["error"]
