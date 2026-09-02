"""Strict, deterministic transaction-cost contracts for MT5 snapshot runs."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd


COST_MODEL_VERSION = "mt5-cost-model/v1"

"""The single canonical specification for explicit MT5 cost models.

Both the parent validator and the public discovery descriptor consume this
object.  Keep behavioural rules here rather than maintaining a tool schema or
documentation-only copy alongside the executable validator.
"""
COST_MODEL_SPEC: dict[str, Any] = {
    "version": COST_MODEL_VERSION,
    "source": "mt5",
    "provenance": (
        "historical_broker_evidence",
        "live_broker_snapshot",
        "broker_static_metadata",
        "user_configured_assumption",
        "explicit_zero_control",
        "derived",
        "unavailable",
    ),
    "modes": {
        "research_control": {
            "allowed_provenance": ("user_configured_assumption", "explicit_zero_control"),
            "description": "Explicit controls or assumptions only; never broker-cost-grounded.",
        },
        "user_assumption": {
            "allowed_provenance": ("user_configured_assumption", "explicit_zero_control"),
            "description": "All enabled costs are explicit user assumptions.",
        },
        "broker_grounded": {
            "allowed_provenance": None,
            "description": "Use only retained broker evidence; unavailable components must be disabled.",
        },
    },
    "components": {
        "spread": {
            "required_fields": ("enabled", "provenance", "method", "unit"),
            "disabled": {"method": "disabled", "unit": "none"},
            "static_methods": ("fixed_engine_pips", "zero"),
            "static_unit": "engine_pips",
            "static_required_fields": ("value", "application"),
            "application": "per_fill_half_spread",
            "historical": {
                "method": "historical_median_mt5_bar_spread",
                "unit": "engine_pips",
                "provenance": "historical_broker_evidence",
                "mode": "broker_grounded",
                "application": "per_fill_half_spread",
                "parent_resolved_fields": ("value", "application", "evidence"),
            },
        },
        "slippage": {
            "required_fields": ("enabled", "provenance", "method", "unit"),
            "disabled": {"method": "disabled", "unit": "none"},
            "static_methods": ("fixed_engine_pips", "zero"),
            "static_unit": "engine_pips",
            "static_required_fields": ("value", "application"),
            "application": "per_fill_adverse",
            "broker_grounded_enabled": False,
        },
        "commission": {
            "required_fields": ("enabled", "provenance", "method", "unit"),
            "disabled": {"method": "disabled", "unit": "none"},
            "static_methods": ("fixed_currency_per_lot_per_side", "zero"),
            "static_unit": "currency_per_lot_per_side",
            "static_required_fields": ("value", "currency", "application"),
            "currency": "USD",
            "application": "per_side",
            "broker_grounded_enabled": False,
        },
        "swap": {
            "required_fields": ("enabled", "provenance", "method", "unit"),
            "disabled": {"method": "disabled", "unit": "none"},
            "method": "fixed_currency_per_lot_per_day",
            "unit": "currency_per_lot_per_day",
            "required_fields_when_enabled": ("value_long", "value_short", "currency", "rollover_basis"),
            "currency": "USD",
            "rollover_basis": "utc_calendar_wednesday_triple",
            "broker_grounded_enabled": False,
        },
    },
    "forbidden_provenance": ("engine_default",),
    "implicit_defaults_allowed": False,
}

MODES = frozenset(COST_MODEL_SPEC["modes"])
PROVENANCE = frozenset(COST_MODEL_SPEC["provenance"])
COMPONENT_NAMES = tuple(COST_MODEL_SPEC["components"])
MAX_HISTORICAL_SPREAD_ZERO_FRACTION = 0.50


class MT5CostModelError(ValueError):
    """Raised when an explicit MT5 cost model is ambiguous or unsupported."""


def _error(message: str) -> MT5CostModelError:
    return MT5CostModelError(f"MT5 transaction-cost contract failure: {message}")


def _disabled_component(name: str, provenance: str) -> dict[str, Any]:
    spec = COST_MODEL_SPEC["components"][name]
    return {
        "enabled": False,
        "method": spec["disabled"]["method"],
        "unit": spec["disabled"]["unit"],
        "provenance": provenance,
    }


def _static_component(name: str, *, value: float, provenance: str) -> dict[str, Any]:
    """Build an example from the canonical component specification."""
    spec = COST_MODEL_SPEC["components"][name]
    component: dict[str, Any] = {
        "enabled": True,
        "method": spec["static_methods"][0],
        "unit": spec["static_unit"],
        "value": value,
        "provenance": provenance,
        "application": spec["application"],
    }
    if "currency" in spec:
        component["currency"] = spec["currency"]
    return component


def _example_model(mode: str) -> dict[str, Any]:
    """Generate a validator-compatible example without a second schema copy."""
    if mode not in MODES:
        raise ValueError(f"unsupported MT5 cost-model mode {mode!r}")
    if mode == "broker_grounded":
        spread_spec = COST_MODEL_SPEC["components"]["spread"]["historical"]
        return {
            "version": COST_MODEL_VERSION,
            "mode": mode,
            "spread": {
                "enabled": True,
                "method": spread_spec["method"],
                "unit": spread_spec["unit"],
                "provenance": spread_spec["provenance"],
            },
            "slippage": _disabled_component("slippage", "unavailable"),
            "commission": _disabled_component("commission", "unavailable"),
            "swap": _disabled_component("swap", "unavailable"),
        }
    provenance = "explicit_zero_control" if mode == "research_control" else "user_configured_assumption"
    return {
        "version": COST_MODEL_VERSION,
        "mode": mode,
        "spread": _static_component("spread", value=0.0, provenance=provenance),
        "slippage": _static_component("slippage", value=0.0, provenance=provenance),
        "commission": _static_component("commission", value=0.0, provenance=provenance),
        "swap": _disabled_component("swap", provenance),
    }


def mt5_cost_model_contract() -> dict[str, Any]:
    """Return the public, read-only contract generated from runtime rules.

    ``examples`` intentionally contain a structural broker-grounded request;
    its historical spread value/evidence is resolved only by the trusted parent
    against the prepared MT5 snapshot.
    """
    components = deepcopy(COST_MODEL_SPEC["components"])
    return {
        "source": COST_MODEL_SPEC["source"],
        "version": COST_MODEL_SPEC["version"],
        "modes": deepcopy(COST_MODEL_SPEC["modes"]),
        "components": components,
        "provenance": list(COST_MODEL_SPEC["provenance"]),
        "forbidden_provenance": list(COST_MODEL_SPEC["forbidden_provenance"]),
        "implicit_defaults_allowed": COST_MODEL_SPEC["implicit_defaults_allowed"],
        "conditional_rules": {
            "broker_grounded": {
                "spread": "must use retained historical_median_mt5_bar_spread evidence",
                "slippage": "must be disabled; broker-grounded bar slippage is unavailable",
                "commission": "must be disabled; generic broker commission schedule is unavailable",
                "swap": "must be disabled; broker rollover modelling is unavailable",
            },
            "disabled_components": "must use method='disabled' and unit='none'",
        },
        "examples": {
            "research_control_explicit_zero": _example_model("research_control"),
            "user_assumption_fixed_costs": {
                **_example_model("user_assumption"),
                "spread": _static_component("spread", value=1.0, provenance="user_configured_assumption"),
                "slippage": _static_component("slippage", value=0.0, provenance="user_configured_assumption"),
                "commission": _static_component("commission", value=2.0, provenance="user_configured_assumption"),
            },
            "broker_grounded_historical_spread": _example_model("broker_grounded"),
        },
    }


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
    for field in COST_MODEL_SPEC["components"][name]["required_fields"]:
        if field not in value:
            raise _error(f"cost_model.{name}.{field} is required")
    return value


def _require_disabled(component: dict[str, Any], name: str) -> None:
    disabled = COST_MODEL_SPEC["components"][name]["disabled"]
    if component.get("method") != disabled["method"] or component.get("unit") != disabled["unit"]:
        raise _error(f"disabled {name} must use method='disabled' and unit='none'")


def _validate_static_component(
    component: dict[str, Any],
    name: str,
    *,
    unit: str,
    allowed_methods: tuple[str, ...],
) -> None:
    if not component["enabled"]:
        _require_disabled(component, name)
        return
    for field in COST_MODEL_SPEC["components"][name]["static_required_fields"]:
        if field not in component:
            raise _error(f"cost_model.{name}.{field} is required")
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
        if provenance in COST_MODEL_SPEC["forbidden_provenance"]:
            raise _error("engine_default is not accepted for explicit MT5 cost models")
        allowed = COST_MODEL_SPEC["modes"][mode]["allowed_provenance"]
        if allowed is not None and provenance not in allowed:
            label = "user/zero" if mode == "user_assumption" else "explicit"
            raise _error(f"{mode} {name} must declare {label} provenance")


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
    elif spread.get("method") == COST_MODEL_SPEC["components"]["spread"]["historical"]["method"]:
        historical = COST_MODEL_SPEC["components"]["spread"]["historical"]
        if model["mode"] != historical["mode"]:
            raise _error("historical MT5 spread is reserved for broker_grounded mode")
        if spread.get("unit") != historical["unit"] or spread.get("provenance") != historical["provenance"]:
            raise _error("historical MT5 spread requires engine_pips and historical_broker_evidence")
        evidence, evidence_frame = _historical_spread_evidence(
            frame,
            point=point,
            engine_pip_size=engine_pip_size,
            requested_symbol=requested_symbol,
            resolved_symbol=resolved_symbol,
        )
        spread["value"] = evidence["value_engine_pips"]
        spread["application"] = historical["application"]
        spread["evidence"] = evidence
    else:
        spread_spec = COST_MODEL_SPEC["components"]["spread"]
        _validate_static_component(
            spread,
            "spread",
            unit=spread_spec["static_unit"],
            allowed_methods=spread_spec["static_methods"],
        )
        if spread.get("application") != spread_spec["application"]:
            raise _error("spread application must be per_fill_half_spread")
        if model["mode"] == "broker_grounded":
            raise _error("broker_grounded spread requires retained historical MT5 bar evidence")

    slippage = model["slippage"]
    slippage_spec = COST_MODEL_SPEC["components"]["slippage"]
    _validate_static_component(
        slippage,
        "slippage",
        unit=slippage_spec["static_unit"],
        allowed_methods=slippage_spec["static_methods"],
    )
    if slippage["enabled"] and slippage.get("application") != slippage_spec["application"]:
        raise _error("slippage application must be per_fill_adverse")
    if model["mode"] == "broker_grounded" and slippage["enabled"] and not slippage_spec["broker_grounded_enabled"]:
        raise _error("broker_grounded slippage is unavailable in MT5 bar mode")

    commission = model["commission"]
    commission_spec = COST_MODEL_SPEC["components"]["commission"]
    _validate_static_component(
        commission,
        "commission",
        unit=commission_spec["static_unit"],
        allowed_methods=commission_spec["static_methods"],
    )
    if commission["enabled"]:
        if commission.get("currency") != commission_spec["currency"] or commission.get("application") != commission_spec["application"]:
            raise _error("commission requires currency='USD' and application='per_side'")
    if model["mode"] == "broker_grounded" and commission["enabled"] and not commission_spec["broker_grounded_enabled"]:
        raise _error("broker commission schedule is unavailable for broker_grounded MT5 runs")

    swap = model["swap"]
    swap_spec = COST_MODEL_SPEC["components"]["swap"]
    if not swap["enabled"]:
        _require_disabled(swap, "swap")
    elif swap.get("method") == swap_spec["method"]:
        if model["mode"] == "broker_grounded" and not swap_spec["broker_grounded_enabled"]:
            raise _error("broker-grounded MT5 swap is unsupported without rollover modeling")
        for field in swap_spec["required_fields_when_enabled"]:
            if field not in swap:
                raise _error(f"cost_model.swap.{field} is required")
        if swap.get("unit") != swap_spec["unit"]:
            raise _error("swap unit must be currency_per_lot_per_day")
        _number(swap.get("value_long"), "cost_model.swap.value_long", minimum=-float("inf"))
        _number(swap.get("value_short"), "cost_model.swap.value_short", minimum=-float("inf"))
        if swap.get("currency") != swap_spec["currency"] or swap.get("rollover_basis") != swap_spec["rollover_basis"]:
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
    if (
        spread["enabled"]
        and spread.get("method") == COST_MODEL_SPEC["components"]["spread"]["historical"]["method"]
    ):
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
    if not (
        spread["enabled"]
        and spread.get("method") == COST_MODEL_SPEC["components"]["spread"]["historical"]["method"]
    ):
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
