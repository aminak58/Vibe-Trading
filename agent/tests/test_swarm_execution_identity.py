"""Identity snapshot and strict preset-capability regression coverage."""

from __future__ import annotations

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
)
from src.swarm.presets import build_run_from_preset
from src.swarm.worker import _validate_worker_execution_identity, build_worker_prompt
from src.swarm.models import SwarmAgentSpec
from src.tools.swarm_tool import SwarmTool, _build_variables


def _verified_mt5_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-gold",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
        ),
        requests=(ExecutionRequest(request_id="gold", symbol="XAUUSD", source="mt5"),),
        resolutions=(
            ExecutionResolution(
                request_id="gold",
                resolved_symbol="XAUUSD_o",
                source="mt5",
                asset_class="forex",
                market="forex",
                resolver_evidence_ref="resolver:1",
            ),
        ),
    )


def test_strict_identity_rejects_generic_quant_strategy_preset() -> None:
    with pytest.raises(ValueError, match="not declared compatible"):
        build_run_from_preset("quant_strategy_desk", {}, execution_identity=_verified_mt5_identity())


def test_strict_identity_snapshots_hash_on_capable_preset() -> None:
    identity = _verified_mt5_identity()

    run = build_run_from_preset("quant_scalp_desk", {"goal": "test", "market": "forex"}, execution_identity=identity)

    assert run.execution_identity == identity
    assert run.identity_hash == identity.identity_hash
    assert run.provenance_validation_status == "pending"
    assert run.preset_capabilities is not None


def test_verified_identity_market_overrides_incidental_usdt_goal() -> None:
    variables = _build_variables(
        "quant_strategy_desk",
        "Historical XAUT-USDT on OKX appears in context.",
        execution_identity=_verified_mt5_identity(),
    )

    assert variables["market"] == "forex"


def test_worker_contract_is_immutable_and_blocks_wrong_source() -> None:
    identity = _verified_mt5_identity()
    prompt = build_worker_prompt(
        SwarmAgentSpec(id="worker", role="role", system_prompt="instructions"),
        {},
        "",
        execution_identity=identity,
    )

    blocked = _validate_worker_execution_identity(
        identity,
        "get_market_data",
        {"codes": ["XAUT-USDT"], "source": "okx"},
    )

    assert "Execution Contract (IMMUTABLE)" in prompt
    assert identity.identity_hash in prompt
    assert blocked is not None
    assert blocked["error_code"] == "identity_blocked"


def test_incident_context_cannot_mutate_verified_mt5_identity() -> None:
    """Historical XAUT/USDT prose is non-authoritative once identity is verified."""
    identity = _verified_mt5_identity()
    incident_prompt = (
        "Run the attached XAUUSD MetaTrader/MQL5 research request. "
        "Historical memory discussed BTCUSD, YM, XAUT-USDT on OKX and yfinance."
    )

    variables = _build_variables("quant_scalp_desk", incident_prompt, execution_identity=identity)
    run = build_run_from_preset("quant_scalp_desk", variables, execution_identity=identity)
    blocked = _validate_worker_execution_identity(
        identity,
        "get_market_data",
        {"codes": ["XAUT-USDT"], "source": "okx"},
    )

    assert variables["market"] == "forex"
    assert run.identity_hash == identity.identity_hash
    assert blocked is not None
    assert blocked["error_code"] == "identity_blocked"


def test_quant_scalp_preset_has_no_hard_coded_execution_identity() -> None:
    from src.swarm.presets import resolve_preset_path

    text = resolve_preset_path("quant_scalp_desk").read_text(encoding="utf-8")
    # Alternative providers may be named only to prohibit them.  These are
    # formerly operational preset defaults and must no longer appear at all.
    for forbidden in ('XAUUSD_o', 'BTCUSD', 'YM', 'XAUT-USDT'):
        assert forbidden not in text
    assert "Execution Contract" in text


def test_public_swarm_schema_cannot_issue_authoritative_identity() -> None:
    assert set(SwarmTool.parameters["properties"]) == {"prompt", "preset_name"}
    response = SwarmTool().execute(
        prompt="XAUUSD source=mt5",
        preset_name="quant_scalp_desk",
    )
    assert "trusted server-side execution identity" in response
