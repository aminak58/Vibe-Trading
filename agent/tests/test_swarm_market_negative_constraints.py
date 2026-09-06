"""Regression coverage for prohibited market/source prose in Swarm goals."""

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
    SyntheticDataPolicy,
)
from src.agent.execution_identity_state import ExecutionIdentityLedger
from src.agent.workflow_obligation import PresetRequirementKind, WorkflowMode, obligation_from_current_user_message
from src.tools.swarm_tool import _build_variables, _extract_market


def _verified_mt5_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="strict-gold",
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
                asset_class="forex",
                market="forex",
                resolver_evidence_ref="resolver:mt5",
            ),
        ),
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "Use XAUUSD MT5. Do not use XAU-USDT or OKX.",
        "Do not use crypto gold or XAUT/PAXG.",
        "no Yahoo, no OKX, no XAUT, no PAXG",
        "XAU-USDT is forbidden.",
    ],
)
def test_english_negative_market_mentions_do_not_select_crypto(prompt: str) -> None:
    assert _extract_market(prompt) != "crypto"


@pytest.mark.parametrize(
    "prompt",
    [
        "برای XAUUSD با داده MT5 بک‌تست بگیر؛ از XAU-USDT یا OKX استفاده نکن.",
        "XAU-USDT مجاز نیست؛ کریپتو استفاده نکن.",
        "طلای توکنی استفاده نکن؛ نه XAUT نه PAXG.",
        "OKX مجاز نیست؛ از Yahoo استفاده نکن.",
    ],
)
def test_persian_negative_market_mentions_do_not_select_crypto(prompt: str) -> None:
    assert _extract_market(prompt) != "crypto"


@pytest.mark.parametrize(
    "prompt",
    [
        "Analyze XAU-USDT on OKX.",
        "Use BTC-USDT for crypto market research.",
        "با BTCUSDT روی OKX بک‌تست بگیر.",
        "از OKX داده بگیر.",
    ],
)
def test_positive_crypto_requests_still_select_crypto(prompt: str) -> None:
    assert _extract_market(prompt) == "crypto"


def test_verified_strict_mt5_identity_overrides_prohibited_crypto_prose() -> None:
    variables = _build_variables(
        "quant_scalp_desk",
        "با استفاده از Swarm این استراتژی XAUUSD را با داده MT5 بررسی کن؛ از XAU-USDT استفاده نکن.",
        execution_identity=_verified_mt5_identity(),
    )

    assert variables["market"] == "forex"


def test_persian_strict_swarm_request_does_not_gain_crypto_from_exclusion_prose(tmp_path) -> None:
    prompt = (
        "با استفاده از Swarm این استراتژی XAUUSD را با داده MT5 بررسی کن؛ "
        "از XAU-USDT یا OKX استفاده نکن."
    )
    obligation = obligation_from_current_user_message(prompt)
    pending_identity = ExecutionIdentityLedger(run_dir=tmp_path, user_message=prompt).snapshot()

    assert obligation.mode is WorkflowMode.SWARM_REQUIRED
    assert obligation.preset_requirement.kind is PresetRequirementKind.CAPABILITY
    assert pending_identity.requests[0].source == "mt5"
    assert _build_variables("quant_scalp_desk", prompt, execution_identity=_verified_mt5_identity())["market"] == "forex"
