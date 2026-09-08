"""Strict server-owned backtest worker input-bundle contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    validate_strict_backtest_package,
)


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
