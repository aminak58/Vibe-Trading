"""Source-aware MT5 identity and fail-closed market-data regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.grounding import GroundingLedger
from src.market_data import fetch_market_data
from src.tools import symbol_search_tool as symbol_search
from src.tools.symbol_search_tool import SymbolSearchTool
from src.agent.execution_identity_state import ExecutionIdentityLedger


def _mt5_result(symbol: str, resolved: str) -> str:
    return json.dumps(
        {
            "ok": True,
            "market": "mt5",
            "source": "symbol_search",
            "data": {
                "query": symbol,
                "count": 1,
                "candidates": [
                    {
                        "symbol": symbol,
                        "requested_symbol": symbol,
                        "resolved_symbol": resolved,
                        "market": "forex",
                        "market_type": "forex",
                        "type": "forex",
                        "exchange": "MT5",
                        "source": "mt5",
                        "source_namespace": "connected_mt5_broker",
                    }
                ],
                "sources": {"mt5": "ok"},
            },
        }
    )


def _patch_mt5(monkeypatch, aliases: dict[str, str]) -> None:
    from backtest.loaders import mt5_loader

    fake = SimpleNamespace(symbol_info=lambda name: SimpleNamespace(description=f"{name} broker instrument"))
    monkeypatch.setattr(mt5_loader, "_import_mt5", lambda: fake)
    monkeypatch.setattr(mt5_loader, "_ensure_initialized", lambda: True)
    monkeypatch.setattr(mt5_loader, "_resolve_broker_symbol", lambda _mt5, code: aliases.get(code.upper()))


def test_mt5_search_resolves_broker_aliases_without_public_search(monkeypatch) -> None:
    _patch_mt5(monkeypatch, {"XAUUSD": "XAUUSD_o", "EURUSD": "EURUSD_o"})
    monkeypatch.setattr(symbol_search, "_search_eastmoney", lambda *_: (_ for _ in ()).throw(AssertionError("public search")))
    monkeypatch.setattr(symbol_search, "_search_yahoo", lambda *_: (_ for _ in ()).throw(AssertionError("public search")))
    monkeypatch.setattr(symbol_search, "_search_selected_connector", lambda *_: (_ for _ in ()).throw(AssertionError("selected connector")))

    xau = json.loads(SymbolSearchTool().execute(query="XAUUSD", source="mt5"))
    eur = json.loads(SymbolSearchTool().execute(query="EURUSD", source="mt5"))

    assert xau["data"]["sources"] == {"mt5": "ok"}
    assert xau["data"]["candidates"][0]["resolved_symbol"] == "XAUUSD_o"
    assert eur["data"]["candidates"][0]["resolved_symbol"] == "EURUSD_o"


def test_unknown_mt5_symbol_is_not_found_without_public_search(monkeypatch) -> None:
    _patch_mt5(monkeypatch, {})
    monkeypatch.setattr(symbol_search, "_search_eastmoney", lambda *_: (_ for _ in ()).throw(AssertionError("public search")))
    monkeypatch.setattr(symbol_search, "_search_yahoo", lambda *_: (_ for _ in ()).throw(AssertionError("public search")))

    payload = json.loads(SymbolSearchTool().execute(query="NOT_A_BROKER_SYMBOL", source="mt5"))
    assert payload["data"]["candidates"] == []
    assert payload["data"]["sources"] == {"mt5": "ok"}


def test_mt5_identity_lock_authorizes_only_mt5_market_data(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="XAUUSD source=mt5")
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "XAUUSD", "source": "mt5"},
        result=_mt5_result("XAUUSD", "XAUUSD_o"),
        call_id="resolve",
        success=True,
    )

    record = ledger.identity_summary()["records"][0]
    assert record["requested_symbol"] == "XAUUSD"
    assert record["resolved_symbol"] == "XAUUSD_o"
    assert record["source"] == ["mt5"]
    assert record["source_namespace"] == "connected_mt5_broker"
    assert record["status"] == "locked"
    assert ledger.authorize_tool_call(
        "get_market_data", {"codes": ["XAUUSD"], "source": "mt5"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="mt5-data",
    ).allowed
    # The provider-native alias remains valid inside the same MT5 namespace.
    assert ledger.authorize_tool_call(
        "get_market_data", {"codes": ["XAUUSD_o"], "source": "mt5"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="mt5-alias",
    ).allowed

    public = ledger.authorize_tool_call(
        "get_market_data", {"codes": ["XAUUSD"], "source": "yahoo"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="public-data",
    )
    connector = ledger.authorize_tool_call(
        "trading_quote", {"symbol": "XAUUSD"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="connector-quote",
    )
    assert public.error_code == "identity_source_mismatch"
    assert connector.error_code == "identity_source_mismatch"


def test_explicit_mt5_user_request_rejects_a_source_blind_resolver(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Research XAUUSD; source = MT5")
    blocked = ledger.authorize_tool_call(
        "search_symbol", {"query": "XAUUSD"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="wrong-resolver",
    )
    assert blocked.allowed is False
    assert blocked.error_code == "identity_source_required"

    allowed = ledger.authorize_tool_call(
        "search_symbol", {"query": "XAUUSD", "source": "mt5"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="mt5-resolver",
    )
    assert allowed.allowed is True


def test_request_without_explicit_source_keeps_generic_resolution(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Research XAUUSD")
    assert ledger.authorize_tool_call(
        "search_symbol", {"query": "XAUUSD"},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="generic-resolver",
    ).allowed


def test_explicit_mt5_market_data_never_uses_forex_fallback() -> None:
    attempts: list[str] = []

    class EmptyLoader:
        def fetch(self, *args, **kwargs):
            return {}

    def resolver(source: str):
        attempts.append(source)
        return EmptyLoader

    result = fetch_market_data(
        codes=["XAUUSD"],
        start_date="2026-08-24",
        end_date="2026-08-24",
        source="mt5",
        loader_resolver=resolver,
    )
    assert attempts == ["mt5"]
    assert result["_unresolved"] == ["XAUUSD"]


def test_explicit_yahoo_keeps_public_symbol_search(monkeypatch) -> None:
    monkeypatch.setattr(symbol_search, "_search_eastmoney", lambda _query: ([], "ok"))
    monkeypatch.setattr(
        symbol_search,
        "_search_yahoo",
        lambda _query: ([{"symbol": "AAPL.US", "name": "Apple", "source": "yahoo"}], "ok"),
    )
    monkeypatch.setattr(symbol_search, "_enrich_us_cik", lambda candidates: (candidates, symbol_search._NO_US))

    payload = json.loads(SymbolSearchTool().execute(query="AAPL", source="yahoo"))
    assert payload["data"]["candidates"][0]["symbol"] == "AAPL.US"
    assert "yahoo" in payload["data"]["sources"]


def test_memory_irrelevance_does_not_change_agentloop_mt5_identity(tmp_path: Path) -> None:
    """The incident invariant: recalled XAUT history cannot set execution state."""
    with_memory = ExecutionIdentityLedger(
        run_dir=tmp_path / "with-memory",
        user_message="Run Swarm for XAUUSD source=mt5; historical XAUT-USDT/OKX note attached.",
    )
    without_memory = ExecutionIdentityLedger(
        run_dir=tmp_path / "without-memory",
        user_message="Run Swarm for XAUUSD source=mt5.",
    )
    assert with_memory.snapshot().requests[0].symbol == "XAUUSD"
    assert with_memory.snapshot().requests[0].source == "mt5"
    assert without_memory.snapshot().requests[0].symbol == "XAUUSD"
    assert without_memory.snapshot().requests[0].source == "mt5"
