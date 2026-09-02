"""Published MT5 backtest-contract parity and discovery tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest.mt5_cost_model import (
    COMPONENT_NAMES,
    COST_MODEL_SPEC,
    COST_MODEL_VERSION,
    MODES,
    PROVENANCE,
    MT5CostModelError,
    mt5_cost_model_contract,
    resolve_cost_model,
)
from src.tools import build_registry, build_swarm_registry
from src.tools.backtest_contract_tool import GetBacktestContractTool


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"spread_points": [10.0, 10.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
    )


def _resolve(model: dict) -> dict:
    resolved, _ = resolve_cost_model(
        model,
        frame=_frame(),
        point=0.00001,
        engine_pip_size=0.0001,
        requested_symbol="EURUSD",
        resolved_symbol="EURUSD_o",
    )
    return resolved


def test_published_contract_uses_runtime_header_enums_and_components() -> None:
    contract = mt5_cost_model_contract()
    assert contract["source"] == "mt5"
    assert contract["version"] == COST_MODEL_VERSION
    assert set(contract["modes"]) == set(MODES)
    assert tuple(contract["components"]) == COMPONENT_NAMES
    assert set(contract["provenance"]) == set(PROVENANCE)
    assert contract["implicit_defaults_allowed"] is False
    assert "engine_default" in contract["forbidden_provenance"]
    assert set(contract["components"]) == set(COST_MODEL_SPEC["components"])
    for name, spec in COST_MODEL_SPEC["components"].items():
        assert contract["components"][name] == spec


@pytest.mark.parametrize(
    "example_name",
    ["research_control_explicit_zero", "user_assumption_fixed_costs", "broker_grounded_historical_spread"],
)
def test_published_valid_examples_pass_runtime_validation(example_name: str) -> None:
    model = mt5_cost_model_contract()["examples"][example_name]
    resolved = _resolve(model)
    assert resolved["resolved"] is True


def test_contract_and_runtime_share_component_method_source_of_truth() -> None:
    contract = mt5_cost_model_contract()
    model = contract["examples"]["user_assumption_fixed_costs"]
    model["spread"]["method"] = "not-in-contract"
    with pytest.raises(MT5CostModelError, match="spread.method is unsupported"):
        _resolve(model)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_each_published_mode_has_a_validator_compatible_example(mode: str) -> None:
    contract = mt5_cost_model_contract()
    example_name = {
        "research_control": "research_control_explicit_zero",
        "user_assumption": "user_assumption_fixed_costs",
        "broker_grounded": "broker_grounded_historical_spread",
    }[mode]
    assert _resolve(contract["examples"][example_name])["mode"] == mode


def test_descriptor_required_components_and_provenance_rules_are_enforced() -> None:
    for component in COMPONENT_NAMES:
        model = mt5_cost_model_contract()["examples"]["research_control_explicit_zero"]
        del model[component]
        with pytest.raises(MT5CostModelError, match=rf"cost_model\.{component} is required"):
            _resolve(model)

    invalid_provenance = mt5_cost_model_contract()["examples"]["research_control_explicit_zero"]
    invalid_provenance["spread"]["provenance"] = "historical_broker_evidence"
    with pytest.raises(MT5CostModelError, match="research_control spread must declare explicit provenance"):
        _resolve(invalid_provenance)

    forbidden_default = mt5_cost_model_contract()["examples"]["research_control_explicit_zero"]
    forbidden_default["spread"]["provenance"] = "engine_default"
    with pytest.raises(MT5CostModelError, match="provenance is unsupported"):
        _resolve(forbidden_default)


def test_broker_grounded_conditional_rules_match_descriptor() -> None:
    contract = mt5_cost_model_contract()
    model = contract["examples"]["broker_grounded_historical_spread"]
    model["slippage"] = {
        "enabled": True,
        "method": "fixed_engine_pips",
        "value": 1.0,
        "unit": "engine_pips",
        "provenance": "historical_broker_evidence",
        "application": "per_fill_adverse",
    }
    assert "must be disabled" in contract["conditional_rules"]["broker_grounded"]["slippage"]
    with pytest.raises(MT5CostModelError, match="broker_grounded slippage is unavailable"):
        _resolve(model)


def test_unsupported_version_and_missing_component_remain_rejected() -> None:
    contract = mt5_cost_model_contract()
    bad_version = contract["examples"]["research_control_explicit_zero"]
    bad_version["version"] = "mt5-cost-model/v999"
    with pytest.raises(MT5CostModelError, match="version is unsupported"):
        _resolve(bad_version)
    missing_component = mt5_cost_model_contract()["examples"]["research_control_explicit_zero"]
    del missing_component["swap"]
    with pytest.raises(MT5CostModelError, match="cost_model.swap is required"):
        _resolve(missing_component)


def test_main_agent_registry_exposes_authoritative_contract() -> None:
    registry = build_registry()
    payload = json.loads(registry.execute("get_backtest_contract", {"source": "mt5"}))
    assert payload["status"] == "ok"
    assert payload["contract"]["version"] == COST_MODEL_VERSION
    # JSON transport converts the descriptor's internal tuples to arrays.
    assert payload["contract"] == json.loads(json.dumps(mt5_cost_model_contract()))


def test_strict_swarm_backtester_can_retrieve_authoritative_contract() -> None:
    registry = build_swarm_registry(["get_backtest_contract"])
    payload = json.loads(registry.execute("get_backtest_contract", {"source": "mt5"}))
    assert payload["status"] == "ok"
    assert payload["contract"]["examples"]["research_control_explicit_zero"]["version"] == COST_MODEL_VERSION


def test_unsupported_discovery_source_is_explicit() -> None:
    payload = json.loads(GetBacktestContractTool().execute(source="yahoo"))
    assert payload["status"] == "error"
    assert payload["error_code"] == "unsupported_backtest_contract_source"
