"""Strict source policy must fail closed while AUTO keeps legacy behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.loaders.base import NoAvailableSourceError
from backtest.runner import fetch_data_map
from src.execution_identity import ExecutionPolicy, FallbackPolicy, SourceMode
from src.market_data import fetch_market_data


def test_strict_market_data_policy_attempts_only_requested_source() -> None:
    attempts: list[str] = []

    class EmptyLoader:
        def fetch(self, *args, **kwargs):
            return {}

    def resolver(source: str):
        attempts.append(source)
        return EmptyLoader

    result = fetch_market_data(
        codes=["EURUSD"],
        start_date="2026-01-01",
        end_date="2026-01-02",
        source="broker_source",
        loader_resolver=resolver,
        execution_policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
        ),
    )

    assert attempts == ["broker_source"]
    assert result["_unresolved"] == ["EURUSD"]


def test_runner_strict_policy_refuses_loader_substitution(monkeypatch) -> None:
    import backtest.runner as runner

    class SubstituteLoader:
        name = "yfinance"

        def fetch(self, *args, **kwargs):
            return {"EURUSD": pd.DataFrame()}

    monkeypatch.setattr(runner, "_get_loader", lambda _source: SubstituteLoader)

    with pytest.raises(NoAvailableSourceError, match="refusing yfinance substitution"):
        fetch_data_map(
            {
                "source": "mt5",
                "codes": ["EURUSD"],
                "interval": "1H",
                "_execution_policy": {"source_mode": "strict"},
            }
        )
