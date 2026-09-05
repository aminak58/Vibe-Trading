"""Package failures must precede MT5 acquisition and strategy execution."""
import json
from pathlib import Path

import pytest

import backtest.mt5_snapshot as snapshots
import src.tools.backtest_tool as tool


@pytest.mark.parametrize("case", ["missing_config", "bad_json", "non_object", "missing_source", "missing_signal", "signal_directory"])
def test_invalid_package_has_no_acquisition_or_config_mutation(tmp_path, monkeypatch, case):
    monkeypatch.setattr(tool, "safe_run_dir", lambda value: tmp_path)
    config_path = tmp_path / "config.json"
    if case != "missing_config":
        content = {"source": "mt5"}
        if case == "non_object":
            content = ["source"]
        elif case == "missing_source":
            content = {}
        config_path.write_text("{" if case == "bad_json" else json.dumps(content))
    if case == "signal_directory":
        (tmp_path / "code" / "signal_engine.py").mkdir(parents=True)
    before = config_path.read_bytes() if config_path.exists() else None
    acquisitions = []
    def acquire(*args, **kwargs):
        acquisitions.append(True)
        raise RuntimeError("Acquisition must not be reached for an invalid package")
    monkeypatch.setattr(snapshots, "prepare_mt5_snapshot", acquire)
    def execute(*args, **kwargs):
        pytest.fail("Invalid package reached strategy runner")
    monkeypatch.setattr(tool.Runner, "execute", execute)
    result = json.loads(tool.run_backtest(str(tmp_path)))
    assert result["status"] == "error"
    assert acquisitions == []
    assert (config_path.read_bytes() if config_path.exists() else None) == before
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "execution_provenance.json").exists()


def test_preflight_does_not_execute_strategy_in_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "safe_run_dir", lambda value: tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"source": "mt5"}))
    (tmp_path / "code").mkdir()
    marker = tmp_path / "parent-executed-strategy"
    (tmp_path / "code" / "signal_engine.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('must stay sandboxed')\n"
    )
    reached = []
    def acquire(*args, **kwargs):
        reached.append(True)
        raise RuntimeError("controlled acquisition stop")
    monkeypatch.setattr(snapshots, "prepare_mt5_snapshot", acquire)
    result = json.loads(tool.run_backtest(str(tmp_path)))
    assert reached == [True]
    assert "controlled acquisition stop" in result["error"]
    assert not marker.exists()
