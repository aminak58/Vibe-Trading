"""Strict parent-to-child MT5 snapshot handoff tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.runner as runner
import src.tools.path_utils as path_utils
import src.tools.backtest_tool as backtest_tool
from backtest.mt5_snapshot import (
    MANIFEST_RELATIVE_PATH,
    MT5SnapshotError,
    load_mt5_snapshot,
    prepare_mt5_snapshot,
)


def _config() -> dict:
    return {
        "codes": ["XAUUSD"],
        "start_date": "2026-08-24",
        "end_date": "2026-08-28",
        "source": "mt5",
        "interval": "5m",
        "engine": "daily",
        "cost_model": _research_control_cost_model(),
    }


def _research_control_cost_model() -> dict:
    return {
        "version": "mt5-cost-model/v1",
        "mode": "research_control",
        "spread": {"enabled": True, "method": "zero", "value": 0.0, "unit": "engine_pips", "provenance": "explicit_zero_control", "application": "per_fill_half_spread"},
        "slippage": {"enabled": True, "method": "zero", "value": 0.0, "unit": "engine_pips", "provenance": "explicit_zero_control", "application": "per_fill_adverse"},
        "commission": {"enabled": True, "method": "zero", "value": 0.0, "unit": "currency_per_lot_per_side", "currency": "USD", "provenance": "explicit_zero_control", "application": "per_side"},
        "swap": {"enabled": False, "method": "disabled", "unit": "none", "provenance": "explicit_zero_control"},
    }


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [2300.0, 2301.0, 2302.0],
            "high": [2301.0, 2302.0, 2303.0],
            "low": [2299.0, 2300.0, 2301.0],
            "close": [2300.5, 2301.5, 2302.5],
            "volume": [10, 11, 12],
        },
        index=pd.date_range("2026-08-24", periods=3, freq="5min", tz="UTC"),
    )


class _DiagnosticMT5:
    TIMEFRAME_M5 = 5

    def __init__(self, *, rates=None, rates_error: Exception | None = None) -> None:
        self._rates = rates
        self._rates_error = rates_error

    def symbol_info(self, _symbol: str):
        return SimpleNamespace(point=0.01)

    def symbol_select(self, _symbol: str, _visible: bool) -> bool:
        return True

    def copy_rates_range(self, _symbol, _timeframe, _start, _end):
        if self._rates_error is not None:
            raise self._rates_error
        return self._rates

    def account_info(self):
        return SimpleNamespace(server="Diagnostic-MT5")


def _diagnostic_rates(*, timestamp: str = "2026-08-24T00:00:00Z", include_close: bool = True):
    row = {
        "time": int(pd.Timestamp(timestamp).timestamp()),
        "open": 2300.0,
        "high": 2301.0,
        "low": 2299.0,
        "tick_volume": 10,
        "spread": 12,
    }
    if include_close:
        row["close"] = 2300.5
    return [row]


def _install_diagnostic_mt5(monkeypatch: pytest.MonkeyPatch, mt5: _DiagnosticMT5) -> None:
    from backtest.loaders import mt5_loader

    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: mt5)
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(mt5_loader, "_resolve_broker_symbol", lambda _mt5, _code: "XAUUSD_o")


def _assert_diagnostic(
    error: MT5SnapshotError,
    *,
    code: str,
    stage: str,
    resolved_symbol: str | None,
) -> None:
    assert error.error_code == code
    assert error.stage == stage
    assert error.details["requested_symbol"] == "XAUUSD"
    assert error.details["codes"] == ["XAUUSD"]
    assert error.details["interval"] == "5m"
    assert error.details["requested_start"] == "2026-08-24"
    assert error.details["requested_end"] == "2026-08-28"
    assert error.details["resolved_symbol"] == resolved_symbol
    assert f"[{code}]" in str(error)


def test_snapshot_reports_terminal_unavailable_with_structured_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backtest.loaders import mt5_loader

    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: None)
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: False)

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_terminal_unavailable",
        stage="terminal_attach",
        resolved_symbol=None,
    )


def test_snapshot_reports_symbol_resolution_failure_with_structured_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backtest.loaders import mt5_loader

    mt5 = _DiagnosticMT5(rates=_diagnostic_rates())
    _install_diagnostic_mt5(monkeypatch, mt5)
    monkeypatch.setattr(mt5_loader, "_resolve_broker_symbol", lambda _mt5, _code: None)

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_symbol_resolution_failed",
        stage="symbol_resolution",
        resolved_symbol=None,
    )


def test_snapshot_reports_copy_rates_exception_with_sanitized_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_diagnostic_mt5(monkeypatch, _DiagnosticMT5(rates_error=RuntimeError("terminal IPC failed")))

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_copy_rates_exception",
        stage="copy_rates_range",
        resolved_symbol="XAUUSD_o",
    )
    assert raised.value.details["exception_type"] == "RuntimeError"
    assert raised.value.details["exception_message"] == "terminal IPC failed"


def test_snapshot_reports_empty_copy_rates_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_diagnostic_mt5(monkeypatch, _DiagnosticMT5(rates=[]))

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_copy_rates_empty",
        stage="copy_rates_range",
        resolved_symbol="XAUUSD_o",
    )
    assert raised.value.details["raw_row_count"] == 0


def test_snapshot_reports_rows_filtered_to_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_diagnostic_mt5(
        monkeypatch,
        _DiagnosticMT5(rates=_diagnostic_rates(timestamp="2026-01-01T00:00:00Z")),
    )

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_rows_filtered_to_zero",
        stage="frame_filter",
        resolved_symbol="XAUUSD_o",
    )
    assert raised.value.details["raw_row_count"] == 1
    assert raised.value.details["filtered_row_count"] == 0


def test_snapshot_reports_invalid_schema_without_changing_readable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_diagnostic_mt5(monkeypatch, _DiagnosticMT5(rates=_diagnostic_rates(include_close=False)))

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_snapshot_invalid_schema",
        stage="frame_schema",
        resolved_symbol="XAUUSD_o",
    )
    assert "MT5-backed data acquisition/handoff failure" in str(raised.value)


def test_snapshot_reports_missing_ohlc_column_as_invalid_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rates = _diagnostic_rates()
    del rates[0]["open"]
    _install_diagnostic_mt5(monkeypatch, _DiagnosticMT5(rates=rates))

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_snapshot_invalid_schema",
        stage="frame_schema",
        resolved_symbol="XAUUSD_o",
    )


def test_snapshot_validation_error_retains_request_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mt5 = _DiagnosticMT5(rates=_diagnostic_rates())
    _install_diagnostic_mt5(monkeypatch, mt5)
    invalid_frame = _bars().copy()
    invalid_frame.index = invalid_frame.index.tz_localize(None)
    from backtest.loaders import mt5_loader

    monkeypatch.setattr(mt5_loader.DataLoader, "_fetch_one", lambda *args, **kwargs: invalid_frame)

    with pytest.raises(MT5SnapshotError) as raised:
        prepare_mt5_snapshot(tmp_path, _config())

    _assert_diagnostic(
        raised.value,
        code="mt5_snapshot_invalid_schema",
        stage="snapshot_validation",
        resolved_symbol="XAUUSD_o",
    )


def test_backtest_tool_exposes_structured_mt5_acquisition_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    _install_diagnostic_mt5(monkeypatch, _DiagnosticMT5(rates=[]))
    monkeypatch.setattr(backtest_tool, "safe_run_dir", lambda path: Path(path))

    result = json.loads(backtest_tool.run_backtest(str(tmp_path)))

    assert result["status"] == "error"
    assert result["error_code"] == "mt5_copy_rates_empty"
    assert result["stage"] == "copy_rates_range"
    assert result["diagnostic"]["requested_symbol"] == "XAUUSD"


@pytest.fixture
def prepared_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from backtest.loaders import mt5_loader

    fake_mt5 = SimpleNamespace(
        account_info=lambda: SimpleNamespace(server="Demo-MT5"),
        symbol_info=lambda symbol: SimpleNamespace(point=0.01),
    )
    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: fake_mt5)
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(mt5_loader, "_resolve_broker_symbol", lambda mt5, code: "XAUUSD_o")
    monkeypatch.setattr(mt5_loader.DataLoader, "_fetch_one", lambda *args, **kwargs: _bars())
    config = _config()
    manifest = prepare_mt5_snapshot(tmp_path, config)
    config["mt5_snapshot_manifest"] = MANIFEST_RELATIVE_PATH.as_posix()
    return tmp_path, config, manifest


def test_parent_prepares_immutable_manifest_with_alias_and_hash(prepared_snapshot):
    run_dir, config, manifest = prepared_snapshot
    assert manifest["source"] == "mt5"
    assert manifest["requested_symbol"] == "XAUUSD"
    assert manifest["resolved_symbol"] == "XAUUSD_o"
    assert manifest["row_count"] == 3
    assert len(manifest["sha256"]) == 64
    assert manifest["timestamp_timezone"] == "UTC"
    assert manifest["timestamp_encoding"] == "unix_epoch_seconds"
    assert manifest["timestamp_representation"] == "pandas-datetimeindex-utc"
    assert manifest["timezone_confidence"] == "official_mt5_sdk_utc"
    assert (run_dir / manifest["snapshot_path"]).is_file()
    frame, loaded = load_mt5_snapshot(run_dir, config)
    assert len(frame) == 3
    assert str(frame.index.tz) == "UTC"
    assert loaded["sha256"] == manifest["sha256"]


def test_snapshot_hash_mismatch_fails_closed(prepared_snapshot):
    run_dir, config, manifest = prepared_snapshot
    (run_dir / manifest["snapshot_path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MT5SnapshotError, match="hash mismatch"):
        load_mt5_snapshot(run_dir, config)


def test_missing_or_invalid_manifest_fails_closed(tmp_path: Path):
    with pytest.raises(MT5SnapshotError, match="missing or invalid"):
        load_mt5_snapshot(tmp_path, _config())
    (tmp_path / "data").mkdir()
    (tmp_path / MANIFEST_RELATIVE_PATH).write_text("[]", encoding="utf-8")
    with pytest.raises(MT5SnapshotError, match="version is unsupported"):
        load_mt5_snapshot(tmp_path, _config())


def test_parent_mt5_unavailable_fails_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from backtest.loaders import mt5_loader

    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: object())
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: False)
    with pytest.raises(MT5SnapshotError, match="unavailable"):
        prepare_mt5_snapshot(tmp_path, _config())
    assert not (tmp_path / MANIFEST_RELATIVE_PATH).exists()


def test_backtest_tool_aborts_before_child_when_parent_acquisition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    monkeypatch.setattr(backtest_tool, "safe_run_dir", lambda path: Path(path))
    import backtest.mt5_snapshot as snapshot

    monkeypatch.setattr(snapshot, "prepare_mt5_snapshot", lambda *args: (_ for _ in ()).throw(MT5SnapshotError("MT5-backed data acquisition/handoff failure: unavailable")))
    monkeypatch.setattr(backtest_tool.Runner, "execute", lambda *args, **kwargs: pytest.fail("child launched"))
    result = json.loads(backtest_tool.run_backtest(str(tmp_path)))
    assert result["status"] == "error"
    assert "MT5-backed data acquisition/handoff failure" in result["error"]


def test_child_uses_snapshot_without_mt5_or_fallback(
    prepared_snapshot, monkeypatch: pytest.MonkeyPatch
):
    run_dir, config, manifest = prepared_snapshot
    from backtest.loaders import mt5_loader

    (run_dir / "code").mkdir()
    (run_dir / "code" / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_RUN_ROOTS", str(run_dir))
    monkeypatch.setattr(path_utils, "safe_run_dir", lambda path: Path(path))
    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: pytest.fail("child initialized MT5"))
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: pytest.fail("child initialized MT5"))
    monkeypatch.setattr(runner, "fetch_data_map", lambda _config: pytest.fail("fallback path called"))
    monkeypatch.setattr(
        runner, "_load_module_from_file", lambda path, name: SimpleNamespace(SignalEngine=type("SignalEngine", (), {}))
    )
    monkeypatch.setattr(runner, "_validate_signal_engine_class", lambda cls: None)
    observed = {}

    class CapturingEngine:
        def run_backtest(self, config, loader, signal_engine, path, **kwargs):
            del signal_engine, path, kwargs
            observed["data"] = loader.fetch(config["codes"], config["start_date"], config["end_date"])
            observed["provenance"] = config["_mt5_snapshot_provenance"]

    monkeypatch.setattr(runner, "_create_market_engine", lambda source, config, codes: CapturingEngine())
    runner.main(run_dir)
    assert list(observed["data"]) == ["XAUUSD"]
    assert observed["provenance"]["resolved_symbol"] == "XAUUSD_o"
    assert observed["provenance"]["sha256"] == manifest["sha256"]


def test_child_rejects_explicit_mt5_without_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "signal_engine.py").write_text("class SignalEngine: pass\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_RUN_ROOTS", str(tmp_path))
    monkeypatch.setattr(path_utils, "safe_run_dir", lambda path: Path(path))
    monkeypatch.setattr(
        runner, "_load_module_from_file", lambda path, name: SimpleNamespace(SignalEngine=type("SignalEngine", (), {}))
    )
    monkeypatch.setattr(runner, "_validate_signal_engine_class", lambda cls: None)
    with pytest.raises(SystemExit):
        runner.main(tmp_path)


def test_non_mt5_fetch_path_is_unchanged(monkeypatch: pytest.MonkeyPatch):
    bars = _bars()

    class Loader:
        name = "yahoo"

        def fetch(self, codes, start_date, end_date, **kwargs):
            return {codes[0]: bars}

    monkeypatch.setattr(runner, "_get_loader", lambda source: Loader)
    result = runner.fetch_data_map(
        {"codes": ["AAPL.US"], "start_date": "2026-08-24", "end_date": "2026-08-28", "source": "yahoo"}
    )
    assert result.source == "yahoo"
    assert list(result.data_map) == ["AAPL.US"]
