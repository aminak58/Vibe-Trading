"""UTC and IANA-session contract tests for strict MT5 snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engines.base import _align, evaluation_start_index
from backtest.loaders.mt5_loader import _rates_to_frame
from backtest.mt5_snapshot import (
    MANIFEST_RELATIVE_PATH,
    MT5SnapshotError,
    load_mt5_snapshot,
    prepare_mt5_snapshot,
)
from backtest.run_card import write_run_card


_RATES_DTYPE = [
    ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
    ("close", "<f8"), ("tick_volume", "<i8"), ("spread", "<i4"), ("real_volume", "<i8"),
]


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.1000, 1.1002, 1.1004],
            "high": [1.1003, 1.1005, 1.1007],
            "low": [1.0998, 1.1000, 1.1002],
            "close": [1.1002, 1.1004, 1.1006],
            "volume": [10, 11, 12],
        },
        index=pd.date_range("2026-08-24", periods=3, freq="5min", tz="UTC"),
    )


def _config() -> dict:
    return {
        "codes": ["EURUSD"],
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


@pytest.fixture
def prepared_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from backtest.loaders import mt5_loader

    class MT5:
        def account_info(self):
            return type("Account", (), {"server": "Demo-MT5"})()

        def symbol_info(self, symbol):
            del symbol
            return type("Symbol", (), {"point": 0.00001})()

    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: MT5())
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(mt5_loader, "_resolve_broker_symbol", lambda mt5, code: "EURUSD_o")
    monkeypatch.setattr(mt5_loader.DataLoader, "_fetch_one", lambda *args, **kwargs: _bars())
    config = _config()
    manifest = prepare_mt5_snapshot(tmp_path, config)
    return tmp_path, config, manifest


def _replace_manifest(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mt5_rate_epoch_is_converted_to_utc_aware_index() -> None:
    rates = np.array(
        [
            (1787529900, 1.1, 1.2, 1.0, 1.15, 10, 0, 0),
            (1787530200, 1.15, 1.25, 1.1, 1.2, 11, 0, 0),
        ],
        dtype=_RATES_DTYPE,
    )
    frame = _rates_to_frame(rates, "2026-08-24", "2026-08-24")
    assert frame is not None
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2026-08-24T00:05:00Z")


def test_snapshot_csv_round_trip_preserves_explicit_utc(prepared_snapshot) -> None:
    run_dir, config, manifest = prepared_snapshot
    csv_text = (run_dir / manifest["snapshot_path"]).read_text(encoding="utf-8")
    assert "+00:00" in csv_text
    frame, loaded = load_mt5_snapshot(run_dir, config)
    assert str(frame.index.tz) == "UTC"
    assert frame.index.equals(_bars().index)
    assert loaded["timestamp_timezone"] == "UTC"
    assert loaded["timestamp_encoding"] == "unix_epoch_seconds"
    assert loaded["timestamp_representation"] == "pandas-datetimeindex-utc"
    assert loaded["timezone_confidence"] == "official_mt5_sdk_utc"


def test_legacy_v1_manifest_is_rejected_without_reinterpretation(prepared_snapshot) -> None:
    run_dir, config, _manifest = prepared_snapshot
    _replace_manifest(run_dir / MANIFEST_RELATIVE_PATH, schema_version="mt5-bar-snapshot/v1")
    with pytest.raises(MT5SnapshotError, match="version is unsupported"):
        load_mt5_snapshot(run_dir, config)


def test_child_rejects_naive_csv_even_if_its_hash_is_updated(prepared_snapshot) -> None:
    run_dir, config, manifest = prepared_snapshot
    snapshot = run_dir / manifest["snapshot_path"]
    naive = _bars().copy()
    naive.index = naive.index.tz_localize(None)
    naive.to_csv(snapshot, index=True, index_label="timestamp", lineterminator="\n")
    _replace_manifest(
        run_dir / MANIFEST_RELATIVE_PATH,
        sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    )
    with pytest.raises(MT5SnapshotError, match="not explicitly UTC encoded"):
        load_mt5_snapshot(run_dir, config)


def test_aware_engine_boundary_and_non_session_ma_signals_match_naive_control() -> None:
    naïve_index = pd.date_range("2026-01-05", periods=32, freq="5min")
    utc_index = naïve_index.tz_localize("UTC")
    close = pd.Series(np.linspace(1.10, 1.12, len(naïve_index)))
    naïve = pd.DataFrame(
        {"open": close.to_numpy(), "high": close.to_numpy(), "low": close.to_numpy(),
         "close": close.to_numpy(), "volume": 1},
        index=naïve_index,
    )
    aware = naïve.copy()
    aware.index = utc_index
    def signals(frame: pd.DataFrame) -> pd.Series:
        return (frame["close"].rolling(3).mean() > frame["close"].rolling(8).mean()).fillna(False).astype(float)

    naïve_dates, _, naïve_targets, _ = _align({"EURUSD": naïve}, {"EURUSD": signals(naïve)}, ["EURUSD"])
    aware_dates, _, aware_targets, _ = _align({"EURUSD": aware}, {"EURUSD": signals(aware)}, ["EURUSD"])
    assert aware_dates.tz is not None and str(aware_dates.tz) == "UTC"
    assert np.array_equal(naïve_dates.asi8, aware_dates.asi8)
    assert naïve_targets["EURUSD"].tolist() == aware_targets["EURUSD"].tolist()
    assert evaluation_start_index({"evaluation_start_date": "2026-01-05"}, aware_dates) == 0


def test_london_session_window_is_dst_safe_at_boundaries() -> None:
    utc = pd.DatetimeIndex(
        [
            "2026-01-05T07:55:00Z", "2026-01-05T08:00:00Z",
            "2026-03-30T06:55:00Z", "2026-03-30T07:00:00Z",
        ]
    )
    london = utc.tz_convert("Europe/London")
    allowed = (london.hour >= 8) & (london.hour < 16)
    assert allowed.tolist() == [False, True, False, True]
    assert london[1].strftime("%H:%M") == "08:00"
    assert london[3].strftime("%H:%M") == "08:00"


def test_new_york_and_tokyo_iana_conversion_contract() -> None:
    winter = pd.Timestamp("2026-01-05T13:00:00Z")
    summer = pd.Timestamp("2026-06-01T13:00:00Z")
    assert winter.tz_convert("America/New_York").strftime("%H:%M") == "08:00"
    assert summer.tz_convert("America/New_York").strftime("%H:%M") == "09:00"
    assert winter.tz_convert("Asia/Tokyo").strftime("%H:%M") == "22:00"
    assert summer.tz_convert("Asia/Tokyo").strftime("%H:%M") == "22:00"


def test_run_card_copies_utc_snapshot_provenance(prepared_snapshot) -> None:
    run_dir, config, manifest = prepared_snapshot
    config["_mt5_snapshot_provenance"] = manifest
    card = write_run_card(run_dir, config, {"total_return": 0.0}, data_sources=["mt5"])
    assert card["mt5_snapshot"]["timestamp_timezone"] == "UTC"
    assert card["mt5_snapshot"]["timestamp_representation"] == "pandas-datetimeindex-utc"
