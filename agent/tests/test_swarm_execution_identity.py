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
from src.tools.swarm_tool import _build_variables


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
