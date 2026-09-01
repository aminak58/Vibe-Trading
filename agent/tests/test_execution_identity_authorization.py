"""Strict identity authorization and resolver integration coverage."""

from __future__ import annotations

import json

from src.agent.execution_identity_state import ExecutionIdentityLedger
from src.agent.grounding import GroundingLedger


def _resolution() -> str:
    return json.dumps(
        {
            "ok": True,
            "data": {
                "candidates": [
                    {
                        "symbol": "XAUUSD",
                        "requested_symbol": "XAUUSD",
                        "resolved_symbol": "XAUUSD_o",
                        "source": "mt5",
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
