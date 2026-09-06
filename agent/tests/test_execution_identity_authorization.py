"""Strict identity authorization and resolver integration coverage."""

from __future__ import annotations

import json

from src.agent.execution_identity_state import ExecutionIdentityLedger
from src.agent.grounding import GroundingLedger


def _resolution(*, resolved_symbol: str = "XAUUSD_o", source: str = "mt5") -> str:
    return json.dumps(
        {
            "ok": True,
            "data": {
                "candidates": [
                    {
                        "symbol": "XAUUSD",
                        "requested_symbol": "XAUUSD",
                        "resolved_symbol": resolved_symbol,
                        "source": source,
                        "source_namespace": "connected_mt5_broker",
                        "market_type": "forex",
                        "type": "forex",
                        "exchange": "MT5",
                    }
                ]
            },
        }
    )


def test_verified_mt5_identity_denies_xaut_okx(tmp_path) -> None:
    execution = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"},
        result=_resolution(),
        call_id="resolver-1",
        success=True,
    )
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="Backtest XAUUSD source=mt5",
        execution_identity=execution.snapshot(),
    )

    denied = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["XAUT-USDT"], "source": "okx"},
        batch_authorized_symbols=set(),
        batch_identity_status="locked",
        call_id="bad-data",
    )
    assert denied.allowed is False
    assert denied.error_code == "denied_by_execution_identity"


def test_verified_mt5_identity_allows_requested_and_resolved_alias(tmp_path) -> None:
    execution = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result=_resolution(), call_id="resolver-1", success=True
    )
    ledger = GroundingLedger(run_dir=tmp_path, user_message="XAUUSD source=mt5", execution_identity=execution.snapshot())

    for symbol in ("XAUUSD", "XAUUSD_o"):
        decision = ledger.authorize_tool_call(
            "get_market_data",
            {"codes": [symbol], "source": "mt5"},
            batch_authorized_symbols=set(),
            batch_identity_status="locked",
            call_id=symbol,
        )
        assert decision.allowed is True


def test_duplicate_identical_mt5_resolution_preserves_verified_identity(tmp_path) -> None:
    """A confirmation lookup must not destroy an already verified MT5 lock."""
    execution = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result=_resolution(), call_id="resolver-1", success=True
    )
    first = execution.snapshot()

    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result=_resolution(), call_id="resolver-2", success=True
    )
    repeated = execution.snapshot()

    assert repeated.status.value == "verified"
    assert repeated.identity_hash == first.identity_hash
    assert repeated.resolutions[0].resolved_symbol == "XAUUSD_o"
    persisted = json.loads((tmp_path / "execution_identity.json").read_text(encoding="utf-8"))
    assert persisted["resolutions"][0]["resolved_symbol"] == "XAUUSD_o"

    ledger = GroundingLedger(run_dir=tmp_path, user_message="XAUUSD source=mt5", execution_identity=repeated)
    assert ledger.authorize_tool_call(
        "run_swarm", {}, batch_authorized_symbols=set(), batch_identity_status="locked", call_id="swarm"
    ).allowed


def test_failed_repeat_does_not_destroy_verified_mt5_identity(tmp_path) -> None:
    """A transient repeat lookup is not material conflicting evidence."""
    execution = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result=_resolution(), call_id="resolver-1", success=True
    )
    verified = execution.snapshot()

    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result="temporary resolver error", call_id="resolver-2", success=False
    )

    repeated = execution.snapshot()
    assert repeated.status.value == "verified"
    assert repeated.identity_hash == verified.identity_hash


def test_conflicting_mt5_resolution_preserves_prior_resolution_with_reason(tmp_path) -> None:
    """A different broker alias is a conflict, not an erased verified record."""
    execution = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result=_resolution(), call_id="resolver-1", success=True
    )
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"},
        result=_resolution(resolved_symbol="XAUUSD.alt"),
        call_id="resolver-2",
        success=True,
    )

    identity = execution.snapshot()
    assert identity.status.value == "conflict"
    assert identity.resolutions[0].resolved_symbol == "XAUUSD_o"
    assert identity.conflicts[-1].field == "resolved_symbol"
    assert identity.conflicts[-1].incoming_value == "XAUUSD.alt"


def test_public_resolver_cannot_replace_verified_mt5_identity(tmp_path) -> None:
    execution = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"}, result=_resolution(), call_id="resolver-1", success=True
    )
    execution.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "yahoo"},
        result=_resolution(resolved_symbol="GC=F", source="yahoo"),
        call_id="resolver-2",
        success=True,
    )

    identity = execution.snapshot()
    assert identity.status.value == "conflict"
    assert identity.resolutions[0].resolved_symbol == "XAUUSD_o"
    assert identity.conflicts[-1].field == "source"
    assert identity.conflicts[-1].incoming_value == "yahoo"


def test_prohibited_symbol_list_does_not_seed_grounding_identity(tmp_path) -> None:
    execution = ExecutionIdentityLedger(
        run_dir=tmp_path,
        user_message="requested symbol: XAUUSD\nsource: mt5",
    )
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message=(
            "requested symbol: XAUUSD\nsource: mt5\n"
            "Do not substitute:\n- XAUT-USDT\n- GC=F\n- PAXG"
        ),
        execution_identity=execution.snapshot(),
    )

    assert "XAUT-USDT" not in ledger.authorized_symbols
    assert "GC=F" not in ledger.authorized_symbols
