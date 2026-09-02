"""Strict, deterministic transaction-cost contracts for MT5 snapshot runs."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd


COST_MODEL_VERSION = "mt5-cost-model/v1"

MODES = frozenset(("research_control", "user_assumption", "broker_grounded"))
PROVENANCE = frozenset((
    "historical_broker_evidence",
    "live_broker_snapshot",
    "broker_static_metadata",
    "user_configured_assumption",
    "explicit_zero_control",
    "derived",
    "unavailable",
))
COMPONENT_NAMES = ("spread", "slippage", "commission", "swap")
MAX_HISTORICAL_SPREAD_ZERO_FRACTION = 0.50


class MT5CostModelError(ValueError):
    """Raised when an explicit MT5 cost model is ambiguous or unsupported."""


def _error(message: str) -> MT5CostModelError:
    return MT5CostModelError(f"MT5 transaction-cost contract failure: {message}")


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise _error(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise _error(f"{label} must be finite and >= {minimum}")
    return result


def _component(model: dict[str, Any], name: str) -> dict[str, Any]:
    value = model.get(name)
    if not isinstance(value, dict):
        raise _error(f"cost_model.{name} is required")
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        raise _error(f"cost_model.{name}.enabled must be boolean")
    provenance = value.get("provenance")
    if provenance not in PROVENANCE:
        raise _error(f"cost_model.{name}.provenance is unsupported")
    return value


def _require_disabled(component: dict[str, Any], name: str) -> None:
    if component.get("method") != "disabled" or component.get("unit") != "none":
        raise _error(f"disabled {name} must use method='disabled' and unit='none'")


def _validate_static_component(
    component: dict[str, Any],
    name: str,
    *,
    unit: str,
    allowed_methods: set[str],
) -> None:
    if not component["enabled"]:
        _require_disabled(component, name)
        return
    if component.get("method") not in allowed_methods:
        raise _error(f"cost_model.{name}.method is unsupported")
    if component.get("unit") != unit:
        raise _error(f"cost_model.{name}.unit must be {unit!r}")
    _number(component.get("value"), f"cost_model.{name}.value")


def _validate_mode_provenance(model: dict[str, Any]) -> None:
    mode = model["mode"]
    for name in COMPONENT_NAMES:
        component = model[name]
        provenance = component["provenance"]
        if provenance == "engine_default":  # defensive even though not in enum
            raise _error("engine_default is not accepted for explicit MT5 cost models")
        if mode == "user_assumption" and provenance not in {
            "user_configured_assumption", "explicit_zero_control",
        }:
            raise _error(f"user_assumption {name} must declare user/zero provenance")
        if mode == "research_control" and provenance not in {
            "user_configured_assumption", "explicit_zero_control",
        }:
            raise _error(f"research_control {name} must declare explicit provenance")


def _historical_spread_evidence(
    frame: pd.DataFrame,
    *,
    point: float,
    engine_pip_size: float,
    requested_symbol: str,
    resolved_symbol: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if "spread_points" not in frame.columns:
        raise _error("historical MT5 spread evidence was not retained by parent acquisition")
    if point <= 0 or engine_pip_size <= 0:
        raise _error("point and engine pip size are required for spread normalization")
    values = pd.to_numeric(frame["spread_points"], errors="coerce")
    available = values.dropna()
    count = int(len(available))
    missing = int(values.isna().sum())
    if count < 2:
        raise _error("historical MT5 spread evidence has fewer than two usable rows")
    zero_count = int((available == 0).sum())
    zero_fraction = zero_count / count
    if zero_fraction > MAX_HISTORICAL_SPREAD_ZERO_FRACTION:
        raise _error(
            "historical MT5 spread evidence is zero-dominated "
            f"({zero_count}/{count}, threshold {MAX_HISTORICAL_SPREAD_ZERO_FRACTION:.0%})"
        )
    if (available < 0).any():
        raise _error("historical MT5 spread evidence contains negative points")

    points_per_engine_pip = engine_pip_size / point
    median_points = float(available.median())
    evidence = {
        "source": "mt5_bar_spread_points",
        "requested_symbol": requested_symbol,
        "resolved_symbol": resolved_symbol,
        "point": point,
        "engine_pip_size": engine_pip_size,
        "points_per_engine_pip": points_per_engine_pip,
        "available_count": count,
        "missing_count": missing,
        "zero_count": zero_count,
        "zero_fraction": zero_fraction,
        "min_points": float(available.min()),
        "median_points": median_points,
        "max_points": float(available.max()),
        "aggregate_method": "median_historical_mt5_bar_spread_points",
        "value_engine_pips": median_points / points_per_engine_pip,
        "actual_start": frame.index[0].isoformat(),
        "actual_end": frame.index[-1].isoformat(),
    }
    evidence_frame = pd.DataFrame({"spread_points": values}, index=frame.index)
    evidence_frame.index.name = "timestamp"
    return evidence, evidence_frame


def resolve_cost_model(
    requested: Any,
    *,
    frame: pd.DataFrame,
    point: float,
    engine_pip_size: float,
    requested_symbol: str,
    resolved_symbol: str,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Validate and freeze a parent-owned MT5 cost model.

    Historical spread uses the median of retained MT5 bar-spread points.  This
    is a deterministic aggregate for the bar engine, not variable-spread or
    tick execution.
    """
    if not isinstance(requested, dict):
        raise _error("cost_model is required for explicit MT5 backtests")
    model = deepcopy(requested)
    if model.get("version") != COST_MODEL_VERSION:
        raise _error("cost_model version is unsupported")
    if model.get("mode") not in MODES:
        raise _error("cost_model mode is unsupported")
    for name in COMPONENT_NAMES:
        _component(model, name)

    spread = model["spread"]
    evidence_frame: pd.DataFrame | None = None
    if not spread["enabled"]:
        _require_disabled(spread, "spread")
    elif spread.get("method") == "historical_median_mt5_bar_spread":
        if model["mode"] != "broker_grounded":
            raise _error("historical MT5 spread is reserved for broker_grounded mode")
        if spread.get("unit") != "engine_pips" or spread.get("provenance") != "historical_broker_evidence":
            raise _error("historical MT5 spread requires engine_pips and historical_broker_evidence")
        evidence, evidence_frame = _historical_spread_evidence(
            frame,
            point=point,
            engine_pip_size=engine_pip_size,
            requested_symbol=requested_symbol,
            resolved_symbol=resolved_symbol,
        )
        spread["value"] = evidence["value_engine_pips"]
        spread["application"] = "per_fill_half_spread"
        spread["evidence"] = evidence
    else:
        _validate_static_component(
            spread,
            "spread",
            unit="engine_pips",
            allowed_methods={"fixed_engine_pips", "zero"},
        )
        if spread.get("application") != "per_fill_half_spread":
            raise _error("spread application must be per_fill_half_spread")
        if model["mode"] == "broker_grounded":
            raise _error("broker_grounded spread requires retained historical MT5 bar evidence")

    slippage = model["slippage"]
    _validate_static_component(
        slippage, "slippage", unit="engine_pips", allowed_methods={"fixed_engine_pips", "zero"}
    )
    if slippage["enabled"] and slippage.get("application") != "per_fill_adverse":
        raise _error("slippage application must be per_fill_adverse")
    if model["mode"] == "broker_grounded" and slippage["enabled"]:
        raise _error("broker_grounded slippage is unavailable in MT5 bar mode")

    commission = model["commission"]
    _validate_static_component(
        commission,
        "commission",
        unit="currency_per_lot_per_side",
        allowed_methods={"fixed_currency_per_lot_per_side", "zero"},
    )
    if commission["enabled"]:
        if commission.get("currency") != "USD" or commission.get("application") != "per_side":
            raise _error("commission requires currency='USD' and application='per_side'")
    if model["mode"] == "broker_grounded" and commission["enabled"]:
        raise _error("broker commission schedule is unavailable for broker_grounded MT5 runs")

    swap = model["swap"]
    if not swap["enabled"]:
        _require_disabled(swap, "swap")
    elif swap.get("method") == "fixed_currency_per_lot_per_day":
        if model["mode"] == "broker_grounded":
            raise _error("broker-grounded MT5 swap is unsupported without rollover modeling")
        if swap.get("unit") != "currency_per_lot_per_day":
            raise _error("swap unit must be currency_per_lot_per_day")
        _number(swap.get("value_long"), "cost_model.swap.value_long", minimum=-float("inf"))
        _number(swap.get("value_short"), "cost_model.swap.value_short", minimum=-float("inf"))
        if swap.get("currency") != "USD" or swap.get("rollover_basis") != "utc_calendar_wednesday_triple":
            raise _error("fixed swap requires USD and utc_calendar_wednesday_triple rollover basis")
    else:
        raise _error("cost_model.swap.method is unsupported")

    _validate_mode_provenance(model)
    model["resolved"] = True
    return model, evidence_frame


def validate_resolved_cost_model(model: Any) -> dict[str, Any]:
    """Validate child-visible resolved model without any broker lookup."""
    if not isinstance(model, dict) or model.get("resolved") is not True:
        raise _error("resolved cost_model is required")
    # Resolution validates all semantics; rerun against an inert frame for
    # non-historical models, and validate historical resolved fields directly.
    if model.get("version") != COST_MODEL_VERSION or model.get("mode") not in MODES:
        raise _error("resolved cost_model header is unsupported")
    for name in COMPONENT_NAMES:
        _component(model, name)
    spread = model["spread"]
    if spread["enabled"] and spread.get("method") == "historical_median_mt5_bar_spread":
        evidence = spread.get("evidence")
        if not isinstance(evidence, dict) or _number(spread.get("value"), "resolved spread value") < 0:
            raise _error("resolved historical spread evidence is invalid")
        for key in ("path", "sha256", "available_count", "zero_fraction", "aggregate_method"):
            if key not in evidence:
                raise _error(f"resolved historical spread evidence is missing {key}")
    else:
        # Static models can be checked by resolving a minimal UTC frame; this
        # keeps child validation independent of MT5 while applying all rules.
        resolve_cost_model(
            model,
            frame=pd.DataFrame({"spread_points": [1, 1]}, index=pd.date_range("2026-01-01", periods=2, tz="UTC")),
            point=1.0,
            engine_pip_size=1.0,
            requested_symbol="resolved",
            resolved_symbol="resolved",
        )
    return model


def validate_spread_evidence_file(
    run_dir: Path, model: dict[str, Any], sha256: Callable[[Path], str]
) -> None:
    """Verify child-visible historical-spread evidence without market access."""
    spread = model["spread"]
    if not (spread["enabled"] and spread.get("method") == "historical_median_mt5_bar_spread"):
        return
    evidence = spread["evidence"]
    path = Path(str(evidence["path"]))
    if path.is_absolute() or ".." in path.parts:
        raise _error("spread evidence path is invalid")
    target = Path(run_dir) / path
    if not target.is_file() or sha256(target) != evidence["sha256"]:
        raise _error("spread evidence hash mismatch or file is missing")
    try:
        frame = pd.read_csv(target)
    except (OSError, ValueError) as exc:
        raise _error("spread evidence cannot be loaded") from exc
    if list(frame.columns) != ["timestamp", "spread_points"] or len(frame) != int(evidence["available_count"]):
        raise _error("spread evidence schema or row count is invalid")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise _error("spread evidence timestamps are invalid")
    values = pd.to_numeric(frame["spread_points"], errors="coerce")
    if values.isna().any() or (values < 0).any():
        raise _error("spread evidence values are invalid")
    count = int(len(values))
    zero_count = int((values == 0).sum())
    zero_fraction = zero_count / count
    expected = {
        "zero_count": zero_count,
        "zero_fraction": zero_fraction,
        "min_points": float(values.min()),
        "median_points": float(values.median()),
        "max_points": float(values.max()),
        "actual_start": timestamps.iloc[0].isoformat(),
        "actual_end": timestamps.iloc[-1].isoformat(),
    }
    if evidence.get("aggregate_method") != "median_historical_mt5_bar_spread_points":
        raise _error("spread evidence aggregate method is unsupported")
    if zero_fraction > MAX_HISTORICAL_SPREAD_ZERO_FRACTION:
        raise _error("spread evidence is zero-dominated")
    for key, actual in expected.items():
        recorded = evidence.get(key)
        if isinstance(actual, float):
            if not isinstance(recorded, (int, float)) or not math.isclose(float(recorded), actual):
                raise _error(f"spread evidence {key} does not match its file")
        elif recorded != actual:
            raise _error(f"spread evidence {key} does not match its file")
