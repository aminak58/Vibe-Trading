"""Authoritative, model-visible contracts for source-specific backtests."""

from __future__ import annotations

import json
from typing import Any

from backtest.mt5_cost_model import mt5_cost_model_contract
from src.agent.tools import BaseTool


class GetBacktestContractTool(BaseTool):
    """Expose runtime-derived configuration contracts before a run is written."""

    name = "get_backtest_contract"
    description = (
        "Return the authoritative structured configuration contract for a backtest source. "
        "Call this before writing config.json for an explicit MT5 backtest; do not guess "
        "the required cost_model. This discovery call performs no market-data or broker lookup."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["mt5"],
                "description": "Backtest source whose authoritative contract is requested.",
            },
        },
        "required": ["source"],
        "additionalProperties": False,
    }
    deterministic = True

    def execute(self, **kwargs: Any) -> str:
        source = kwargs.get("source")
        if not isinstance(source, str) or source.casefold() != "mt5":
            return json.dumps(
                {
                    "status": "error",
                    "error_code": "unsupported_backtest_contract_source",
                    "error": "Only the explicit MT5 backtest contract is currently published.",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "source": "mt5",
                "contract": mt5_cost_model_contract(),
            },
            ensure_ascii=False,
        )
