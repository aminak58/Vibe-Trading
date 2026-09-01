"""Strict report provenance validation tests."""

from __future__ import annotations

from src.execution_identity import (
    ExecutionIdentity, ExecutionIdentityStatus, ExecutionMode, ExecutionPolicy,
    ExecutionRequest, ExecutionResolution, FallbackPolicy, SourceMode,
)
from src.execution_provenance import validate_execution_provenance
from src.agent.loop import AgentLoop


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="id", mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.VERIFIED,
        policy=ExecutionPolicy(source_mode=SourceMode.STRICT, fallback=FallbackPolicy.DENY, cross_source_fallback=False),
        requests=(ExecutionRequest(request_id="r", symbol="XAUUSD", source="mt5"),),
        resolutions=(ExecutionResolution(request_id="r", resolved_symbol="XAUUSD_o", source="mt5", resolver_evidence_ref="r"),),
    )


def test_strict_provenance_requires_matching_manifest_fields() -> None:
    identity = _identity()
    result = validate_execution_provenance(
        identity,
        {"identity_hash": identity.identity_hash, "owning_run_id": "swarm-1", "requested_symbol": "XAUUSD", "resolved_symbol": "XAUUSD_o", "effective_source": "mt5", "synthetic": False, "fallback_used": False},
        owning_run_id="swarm-1",
    )
    assert result.status == "passed"


def test_strict_provenance_conflict_invalidates_quantitative_verdict() -> None:
    identity = _identity()
    result = validate_execution_provenance(identity, {"identity_hash": "wrong", "effective_source": "yfinance", "synthetic": True}, owning_run_id="swarm-1")
    assert result.status == "provenance_conflict"
    assert "identity_hash mismatch" in result.issues


def test_active_swarm_blocks_replacement_market_workflow() -> None:
    loop = object.__new__(AgentLoop)
    loop._active_swarm_run_id = "swarm-active"
    loop._active_swarm_identity_hash = "identity-hash"

    blocked = loop._swarm_ownership_block("backtest")
    allowed = loop._swarm_ownership_block("get_swarm_status")

    assert blocked is not None
    assert "swarm_ownership_active" in blocked
    assert allowed is None
