"""Component tests for current-task attachment identity candidates."""

from __future__ import annotations

import json

from src.agent.execution_identity_state import ExecutionIdentityLedger
from src.execution_identity import ExecutionMode, ExecutionIdentityStatus, SourceMode


def test_raw_user_source_creates_strict_partial_identity(tmp_path) -> None:
    ledger = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Backtest XAUUSD source=mt5")

    assert ledger.identity.mode is ExecutionMode.SOURCE_SCOPED
    assert ledger.identity.policy.source_mode is SourceMode.STRICT
    assert ledger.identity.requests[0].symbol == "XAUUSD"


def test_declared_attachment_requirement_enriches_identity(tmp_path) -> None:
    ledger = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Run the attached strategy")
    result = json.dumps(
        {
            "status": "ok",
            "file": "uploads/strategy.txt",
            "text": "Required symbol: XAUUSD\nSource: mt5\nPlatform: MetaTrader 5",
        }
    )

    ledger.ingest_document_result(result, call_id="doc-1")

    assert ledger.identity.mode is ExecutionMode.SOURCE_SCOPED
    assert ledger.identity.status is ExecutionIdentityStatus.PARTIALLY_SPECIFIED
    assert ledger.identity.requests[0].symbol == "XAUUSD"
    assert ledger.identity.requests[0].source == "mt5"
    assert ledger.identity.requests[0].platform == "metatrader 5"
    assert (tmp_path / "execution_identity.json").exists()


def test_unlabelled_attachment_mentions_are_not_execution_requirements(tmp_path) -> None:
    ledger = ExecutionIdentityLedger(run_dir=tmp_path, user_message="Review the attachment")
    result = json.dumps(
        {
            "status": "ok",
            "file": "uploads/history.txt",
            "text": "A historical comparison used XAUT-USDT on OKX.",
        }
    )

    ledger.ingest_document_result(result, call_id="doc-2")

    assert ledger.identity.mode is ExecutionMode.GENERIC
    assert ledger.identity.requests == ()
