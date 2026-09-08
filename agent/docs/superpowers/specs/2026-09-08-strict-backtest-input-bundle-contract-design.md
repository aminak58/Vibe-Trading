# Strict Backtest Input Bundle Contract v0.1

## Purpose

Strict, source-scoped Swarm backtests must receive an immutable, server-owned
package contract before a worker can invoke `backtest()`. The contract removes
model authority to infer the resolved symbol field, file layout, or strategy
document availability.

## Scope

This change is limited to strict `task-backtest` worker handoff and its
pre-backtest validation. It does not change MT5 acquisition, snapshots,
artifact promotion, risk/report tasks, parent orchestration, or strategy
logic.

## Input bundle

The runtime creates a bundle only for source-scoped strict backtest tasks. It
contains:

- requested and resolved symbols;
- the canonical `codes` list containing the verified resolved symbol;
- source and strict fallback/synthetic policy;
- canonical relative paths: `config.json` and `code/signal_engine.py`;
- research-window metadata;
- a strategy-source status and, only when server-owned and materializable, a
  document reference, hash, or extracted text.

The bundle is immutable from the worker's perspective. Content hashes may
support debugging, but do not create authority for an unverified document or
window.

## Preflight

Immediately before a strict worker invokes `backtest()`, the runtime validates
the worker package against the bundle. It requires valid JSON, exact `codes`,
`source=mt5`, valid start/end dates, and the canonical strategy path. If the
task requires a strategy document, an unavailable document produces
`required_strategy_source_unavailable`; it may not be replaced with model
prose.

Preflight failures return one structured error and prevent repeated blind
`backtest({})` calls. They do not call MT5 or generate official artifacts.

## Data flow

```
ExecutionIdentity + trusted document reference + window metadata
  -> server-owned strict input bundle
  -> task-backtest worker prompt/context and artifact workspace
  -> package preflight
  -> BacktestTool / MT5 snapshot acquisition
```

Only a passing preflight may reach MT5 acquisition. A successful backtest still
must satisfy the existing official artifact contract.

## Compatibility and safety

Non-strict workflows retain their current package behavior. Missing legacy
bundle metadata is treated as absent rather than silently upgraded. No public
source fallback, synthetic data, retry, or replacement Swarm is introduced.

## Verification plan

TDD coverage will prove bundle identity derivation, exact symbol schema,
canonical path enforcement, document unavailable fail-fast, date validation,
one-shot preflight blocking, valid mocked acquisition reachability, and
non-strict compatibility. Tests must not call live MT5, providers, or a Swarm
UI.
