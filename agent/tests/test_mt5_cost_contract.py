"""Strict explicit MT5 transaction-cost contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.engines.forex import ForexEngine
from backtest.mt5_cost_model import COST_MODEL_VERSION, MT5CostModelError, resolve_cost_model
from backtest.mt5_snapshot import MT5SnapshotError, load_mt5_snapshot, prepare_mt5_snapshot


def _frame(*, symbol: str = "EURUSD", spreads: list[float] | None = None) -> pd.DataFrame:
    values = spreads if spreads is not None else [10.0, 10.0, 10.0]
    count = len(values)
    base = 1.10 if symbol == "EURUSD" else 2300.0
    return pd.DataFrame(
        {
            "open": [base + i * 0.001 for i in range(count)],
            "high": [base + 0.002 + i * 0.001 for i in range(count)],
            "low": [base - 0.001 + i * 0.001 for i in range(count)],
            "close": [base + 0.001 + i * 0.001 for i in range(count)],
            "volume": [100] * count,
            "spread_points": values,
        },
        index=pd.date_range("2026-08-24", periods=count, freq="5min", tz="UTC"),
    )


def _model(mode: str = "research_control") -> dict:
    zero_provenance = "explicit_zero_control" if mode == "research_control" else "user_configured_assumption"
    return {
        "version": COST_MODEL_VERSION,
        "mode": mode,
        "spread": {"enabled": True, "method": "zero", "value": 0.0, "unit": "engine_pips", "provenance": zero_provenance, "application": "per_fill_half_spread"},
        "slippage": {"enabled": True, "method": "zero", "value": 0.0, "unit": "engine_pips", "provenance": zero_provenance, "application": "per_fill_adverse"},
        "commission": {"enabled": True, "method": "zero", "value": 0.0, "unit": "currency_per_lot_per_side", "currency": "USD", "provenance": zero_provenance, "application": "per_side"},
        "swap": {"enabled": False, "method": "disabled", "unit": "none", "provenance": zero_provenance},
    }


def _resolve(model: dict, *, symbol: str = "EURUSD", spreads: list[float] | None = None) -> dict:
    point = 0.00001 if symbol == "EURUSD" else 0.01
    pip = 0.0001 if symbol == "EURUSD" else 0.10
    resolved, _ = resolve_cost_model(
        model,
        frame=_frame(symbol=symbol, spreads=spreads),
        point=point,
        engine_pip_size=pip,
        requested_symbol=symbol,
        resolved_symbol=f"{symbol}_o",
    )
    return resolved


def test_missing_cost_model_fails_explicit_mt5_resolution() -> None:
    with pytest.raises(MT5CostModelError, match="cost_model is required"):
        resolve_cost_model(None, frame=_frame(), point=0.00001, engine_pip_size=0.0001, requested_symbol="EURUSD", resolved_symbol="EURUSD_o")


def test_research_control_explicit_zeros_are_accepted_without_defaults() -> None:
    resolved = _resolve(_model())
    engine = ForexEngine({"source": "mt5", "cost_model": resolved})
    engine._active_symbol = "EURUSD"
    assert engine.apply_slippage(1.1, 1) == 1.1
    assert engine.calc_commission(100_000, 1.1, 1, True) == 0.0
    assert engine.swap_enabled is False


def test_user_assumption_fixed_spread_and_commission_apply_per_side() -> None:
    model = _model("user_assumption")
    model["spread"] = {"enabled": True, "method": "fixed_engine_pips", "value": 1.0, "unit": "engine_pips", "provenance": "user_configured_assumption", "application": "per_fill_half_spread"}
    model["commission"] = {"enabled": True, "method": "fixed_currency_per_lot_per_side", "value": 2.0, "unit": "currency_per_lot_per_side", "currency": "USD", "provenance": "user_configured_assumption", "application": "per_side"}
    engine = ForexEngine({"source": "mt5", "cost_model": _resolve(model)})
    engine._active_symbol = "EURUSD"
    assert engine.apply_slippage(1.1, 1) == pytest.approx(1.10005)
    assert engine.calc_commission(100_000, 1.1, 1, True) == pytest.approx(2.0)
    assert engine.calc_commission(100_000, 1.1, -1, False) == pytest.approx(2.0)


def test_ambiguous_spread_unit_is_rejected() -> None:
    model = _model("user_assumption")
    model["spread"].update({"method": "fixed_engine_pips", "value": 1.0, "unit": "points", "provenance": "user_configured_assumption"})
    with pytest.raises(MT5CostModelError, match="unit"):
        _resolve(model)


@pytest.mark.parametrize(
    ("symbol", "point", "spreads", "expected_pips"),
    [("EURUSD", 0.00001, [10, 20, 30], 2.0), ("XAUUSD", 0.01, [12, 22, 51], 2.2)],
)
def test_mt5_points_normalize_to_engine_pips(symbol, point, spreads, expected_pips) -> None:
    del point
    model = _model("broker_grounded")
    model["spread"] = {"enabled": True, "method": "historical_median_mt5_bar_spread", "unit": "engine_pips", "provenance": "historical_broker_evidence"}
    model["slippage"] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    model["commission"] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    model["swap"] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    resolved = _resolve(model, symbol=symbol, spreads=spreads)
    assert resolved["spread"]["value"] == pytest.approx(expected_pips)
    assert resolved["spread"]["evidence"]["points_per_engine_pip"] == pytest.approx(10.0)


def test_zero_dominated_historical_spread_is_rejected_for_broker_grounded() -> None:
    model = _model("broker_grounded")
    model["spread"] = {"enabled": True, "method": "historical_median_mt5_bar_spread", "unit": "engine_pips", "provenance": "historical_broker_evidence"}
    for name in ("slippage", "commission", "swap"):
        model[name] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    with pytest.raises(MT5CostModelError, match="zero-dominated"):
        _resolve(model, spreads=[0] * 277 + [9] * 11)


def test_broker_grounded_commission_and_swap_cannot_fall_back() -> None:
    model = _model("broker_grounded")
    model["spread"] = {"enabled": True, "method": "historical_median_mt5_bar_spread", "unit": "engine_pips", "provenance": "historical_broker_evidence"}
    model["slippage"] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    model["commission"] = {"enabled": True, "method": "fixed_currency_per_lot_per_side", "value": 2.0, "unit": "currency_per_lot_per_side", "currency": "USD", "provenance": "broker_static_metadata", "application": "per_side"}
    with pytest.raises(MT5CostModelError, match="commission schedule"):
        _resolve(model, spreads=[12, 22, 51])
    model = _model("broker_grounded")
    model["spread"] = {"enabled": True, "method": "historical_median_mt5_bar_spread", "unit": "engine_pips", "provenance": "historical_broker_evidence"}
    model["slippage"] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    model["commission"] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    model["swap"] = {"enabled": True, "method": "fixed_currency_per_lot_per_day", "value_long": -1.0, "value_short": 1.0, "unit": "currency_per_lot_per_day", "currency": "USD", "provenance": "broker_static_metadata", "rollover_basis": "utc_calendar_wednesday_triple"}
    with pytest.raises(MT5CostModelError, match="swap"):
        _resolve(model, spreads=[12, 22, 51])


def test_parent_freezes_historical_spread_evidence_and_child_verifies_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backtest.loaders import mt5_loader

    model = _model("broker_grounded")
    model["spread"] = {"enabled": True, "method": "historical_median_mt5_bar_spread", "unit": "engine_pips", "provenance": "historical_broker_evidence"}
    for name in ("slippage", "commission", "swap"):
        model[name] = {"enabled": False, "method": "disabled", "unit": "none", "provenance": "unavailable"}
    config = {"codes": ["XAUUSD"], "start_date": "2026-08-24", "end_date": "2026-08-28", "source": "mt5", "interval": "5m", "cost_model": model}
    fake = SimpleNamespace(account_info=lambda: SimpleNamespace(server="Demo"), symbol_info=lambda symbol: SimpleNamespace(point=0.01))
    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: fake)
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(mt5_loader, "_resolve_broker_symbol", lambda mt5, code: "XAUUSD_o")
    monkeypatch.setattr(mt5_loader.DataLoader, "_fetch_one", lambda *args, **kwargs: _frame(symbol="XAUUSD", spreads=[12, 22, 51]))
    manifest = prepare_mt5_snapshot(tmp_path, config)
    evidence = manifest["cost_model"]["spread"]["evidence"]
    assert evidence["value_engine_pips"] == pytest.approx(2.2)
    assert (tmp_path / evidence["path"]).is_file()
    frame, loaded = load_mt5_snapshot(tmp_path, config)
    assert len(frame) == 3
    assert loaded["cost_model"] == config["cost_model"]
    (tmp_path / evidence["path"]).write_text("timestamp,spread_points\n", encoding="utf-8")
    with pytest.raises(MT5SnapshotError, match="spread evidence"):
        load_mt5_snapshot(tmp_path, config)
