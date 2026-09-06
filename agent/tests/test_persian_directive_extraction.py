"""Regression coverage for current-user Persian strict workflow directives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.execution_identity_state import ExecutionIdentityLedger
from src.agent.grounding import GroundingLedger
from src.agent.workflow_obligation import (
    PresetRequirementKind,
    WorkflowMode,
    obligation_from_current_user_message,
)
from src.execution_identity import (
    ExecutionIdentityStatus,
    ExecutionMode,
    FallbackPolicy,
    SourceMode,
    SyntheticDataPolicy,
)


_PERSIAN_STRICT_SWARM_REQUEST = (
    "لطفا با استفاده از Swarm این استراتژی XAUUSD را با داده MT5 ارزیابی و بک‌تست کن"
)


def test_persian_current_user_request_creates_source_scoped_swarm_requirement() -> None:
    obligation = obligation_from_current_user_message(_PERSIAN_STRICT_SWARM_REQUEST)

    assert obligation.mode is WorkflowMode.SWARM_REQUIRED
    assert obligation.authority == "current_user"
    assert obligation.preset_requirement.kind is PresetRequirementKind.CAPABILITY
    assert obligation.preset_requirement.source_scoped_execution is True
    assert obligation.requested_preset is None
    assert obligation.selected_preset is None


@pytest.mark.parametrize(
    "message",
    [
        "با استفاده از سوارم این درخواست را انجام بده",
        "با swarm ارزیابی کن",
        "از swarm استفاده کن",
        "با سوارم ارزیابی کن",
    ],
)
def test_persian_swarm_phrases_create_required_obligation(message: str) -> None:
    assert obligation_from_current_user_message(message).mode is WorkflowMode.SWARM_REQUIRED


def test_quoted_persian_swarm_example_does_not_create_obligation() -> None:
    obligation = obligation_from_current_user_message(
        'یادداشت تاریخی گفت "با استفاده از Swarm این درخواست را انجام بده".'
    )

    assert obligation.mode is WorkflowMode.NONE


@pytest.mark.parametrize(
    "message",
    [
        "XAUUSD با داده MT5",
        "XAUUSD با دادهٔ MT5",
        "XAUUSD با داده متاتریدر",
        "XAUUSD با داده متاتریدر ۵",
        "XAUUSD با داده MetaTrader",
        "از MT5 برای XAUUSD استفاده کن",
    ],
)
def test_persian_mt5_source_phrases_create_strict_pending_identity(message: str, tmp_path: Path) -> None:
    identity = ExecutionIdentityLedger(run_dir=tmp_path, user_message=message).snapshot()

    assert identity.mode is ExecutionMode.SOURCE_SCOPED
    assert identity.status is ExecutionIdentityStatus.PARTIALLY_SPECIFIED
    assert identity.requests[0].symbol == "XAUUSD"
    assert identity.requests[0].source == "mt5"
    assert identity.policy.source_mode is SourceMode.STRICT
    assert identity.policy.fallback is FallbackPolicy.DENY
    assert identity.policy.synthetic is SyntheticDataPolicy.FORBID


def test_persian_strict_identity_becomes_verified_only_from_mt5_resolver(
    tmp_path: Path,
) -> None:
    ledger = ExecutionIdentityLedger(run_dir=tmp_path, user_message=_PERSIAN_STRICT_SWARM_REQUEST)
    ledger.ingest_resolver_result(
        arguments={"query": "XAUUSD", "source": "mt5"},
        result=json.dumps(
            {
                "data": {
                    "candidates": [
                        {
                            "requested_symbol": "XAUUSD",
                            "resolved_symbol": "XAUUSD_o",
                            "source": "mt5",
                            "source_namespace": "connected_mt5_broker",
                            "exchange": "MT5",
                            "type": "forex",
                            "market_type": "forex",
                        }
                    ]
                }
            }
        ),
        call_id="resolver-1",
        success=True,
    )

    identity = ledger.snapshot()
    assert identity.status is ExecutionIdentityStatus.VERIFIED
    assert identity.resolutions[0].resolved_symbol == "XAUUSD_o"
    assert identity.resolutions[0].source == "mt5"


def test_grounding_uses_the_same_persian_current_user_source_intent(tmp_path: Path) -> None:
    grounding = GroundingLedger(run_dir=tmp_path, user_message=_PERSIAN_STRICT_SWARM_REQUEST)

    assert grounding.identity_summary()["requested_source"] == "mt5"


def test_english_source_and_swarm_extraction_remains_unchanged(tmp_path: Path) -> None:
    obligation = obligation_from_current_user_message("Run this with Swarm using XAUUSD source=mt5")
    identity = ExecutionIdentityLedger(
        run_dir=tmp_path, user_message="Run this with Swarm using XAUUSD source=mt5"
    ).snapshot()

    assert obligation.mode is WorkflowMode.SWARM_REQUIRED
    assert identity.mode is ExecutionMode.SOURCE_SCOPED
    assert identity.requests[0].source == "mt5"
