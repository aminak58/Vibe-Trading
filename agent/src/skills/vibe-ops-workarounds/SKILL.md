---
name: vibe-ops-workarounds
description: Known Vibe-Trading infra bugs + safe workarounds (identity gate, get_market_data vs backtest, position_adjustment, Jina read_url 401). No main-code changes — teaches agents to route around them.
category: ops
---

# Vibe-Trading Known Workarounds

> **Purpose:** agents loading this skill learn the known infra limitations and the **safe path** that avoids each bug *without modifying main code*.

## Bug 1 — Identity gate (symbol poisoning)

### Symptom
`search_symbol("YM")` can lock the session identity to an unintended public-catalog symbol (e.g. `YMM.US` = Full Truck Alliance!) instead of `YM=F` (Dow futures). After that, every `get_market_data(..., source="mt5")` fails with `identity_mismatch`.

### Why
- The resolver only matches the **public catalog** (Yahoo/Eastmoney). Broker-local symbols (`XAUUSD_o`, `BTCUSD`, `YM`) do not exist there.
- Once the session identity is locked, silent suffix/exchange rewrites are forbidden.

### Workaround
- **Do not call `get_market_data` for broker-symbol research.** It is identity-gated and cannot reach broker data.
- Use **`backtest` with `"source": "mt5"` + broker codes (`XAUUSD_o`, `BTCUSD`, `YM`): the backtest engine fetches broker data directly and bypasses the identity gate.
- If the agent sees `identity_required`/`identity_mismatch`: do NOT retry — call `search_symbol("YM=F")` (or new session), and prefer **`hold` position_adjustment** for signal-flip scalping (not `rebalance` — that triggers full-notional flips and `insufficient capital`).

## Bug 2 — `get_market_data` vs `backtest`: broker data unreachable

### Why
`get_market_data` has an identity gate; **broker data exists** but only `backtest` can reach it (it fetches internally via `mt5_loader` from the broker terminal).

### Workaround
- For **real backtest metrics**: write `config.json` (`source: "mt5"`, one code per run `["XAUUSD_o"]`, `interval: "1H"` or `"1D"`, `initial_cash`, `commission`, `position_adjustment: "hold"`) and run **`backtest`**.
- Multi-instrument baskets fail on settlement-currency mismatch (e.g. CNY/USD); run **per-symbol** — same `signal_engine.py`.
- **Confirmed by probe:** `XAUUSD_o` (LiteFinance MT5, since 2004), `BTCUSD`, `YM` → backtest **works**, returns real broker metrics (gold spot proxy e.g. `GC=F` / `XAUUSD_o`).

## Bug 3 — Jina `read_url` blocked for this network

### Symptom
`read_url` → HTTP 401 / AS9009 bad-reputation anonymous block (even from Iran with/without VPN).

### Workaround
1. The agent can **read algorithms / logic from the uploaded attachment** instead of fetching PineScript source from the internet.
2. If a URL is genuinely needed: use `web_search` first; if read_url fails, mark "unavailable: <error>" and continue — never fabricate content.

## Acknowledge Tools
- Always create **backups** before editing any code/config.
- Always use **new session** after identity lock poisoning.
- When in doubt: report `unavailable: <error>` + ask the lead.