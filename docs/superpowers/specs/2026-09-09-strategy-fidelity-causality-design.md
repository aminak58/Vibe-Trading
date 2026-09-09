# Strategy Fidelity and Causality Design

**Status:** Proposed for implementation planning after user review
**Program:** Strict Research Harness Reliability
**Phase:** Plan 1 of 5
**Date:** 2026-09-09

## Goal

Make a strict strategy backtest eligible for research interpretation only when the server can prove both of the following:

1. the executable strategy faithfully implements a declared, authoritative strategy contract; and
2. every signal available at time `t` depends only on information available at or before `t`.

The server, not worker prose, owns both determinations. A mechanically successful backtest may still complete for smoke-testing when either gate is not satisfied, but it must be labeled non-authoritative and must not authorize a strategy verdict.

## Why This Phase Exists

Run `swarm-20260908-231853-f91c2579` proved that execution provenance and artifact hashes alone are insufficient:

- the official package was built and executed under the correct `XAUUSD_o / mt5 / strict` identity;
- the executable used D1 EMA, H1 ADX, H4 structure, and M5 VWAP rather than the PDF's H1/M15/M5 hierarchy;
- required concepts such as M15 Order Block/FVG and the complete risk controls were omitted or substituted without structured disclosure;
- higher-timeframe values were constructed by resampling complete D1/H1/H4 bins and forward-filling them into bars inside the same bin, allowing future values to influence earlier M5 decisions;
- report prose later described a different SMA/z-score strategy not found in the executed code.

This phase closes the first two integrity gaps. Cost reconciliation, coverage/sufficiency, risk/report authority, and UI delivery remain separate later phases.

## Design Principles

1. **Agents propose; server contracts authorize.** Worker text is advisory and cannot establish fidelity or causality.
2. **Executable behavior outranks prose.** Docstrings, summaries, and prompts cannot prove implementation.
3. **Unknown fails closed.** Missing evidence yields `unknown` or `not_verified`, never an upgraded PASS.
4. **Mechanical execution is distinct from research authorization.** Smoke tests may run, but downstream verdict strength is gated.
5. **One failure must remain diagnosable.** Fidelity and causality failures use separate codes and structured details.
6. **Legacy artifacts remain readable.** Missing new metadata downgrades authority without crashing.
7. **No strategy redesign in this phase.** The phase validates declared fidelity and causality; it does not decide which trading rules are profitable.

## Scope

### In scope

- a versioned, server-owned strategy contract artifact;
- a versioned implementation-fidelity matrix tied to the executed strategy hash;
- deterministic validation of required, simplified, omitted, substituted, and unknown rules;
- a causal higher-timeframe alignment interface;
- prefix-invariance and closed-bar tests;
- propagation of fidelity and causality status into the official run card and canonical result;
- a research-authorization gate consumed by later risk/report phases;
- offline fixtures derived from the problematic run;
- Cline-as-implementer / Codex-as-gatekeeper execution governance.

### Out of scope

- choosing optimal strategy parameters;
- deciding whether the PDF strategy has an edge;
- improving PDF extraction quality;
- designing a complete PDF-to-code compiler or trading DSL;
- MT5 history discovery and sample thresholds;
- transaction-cost and equity reconciliation;
- risk-auditor statistical policy;
- final report prose enforcement;
- parent/session/UI lifecycle changes;
- live Swarm, MT5, provider, or UI execution during implementation.

## Architecture

The phase adds two deterministic gates between authoritative strategy-source handoff and research interpretation:

```text
Authoritative strategy source
        |
        v
Server-owned StrategyContract
        |
        v
Executed strategy package + hashes
        |
        +--> Fidelity validator --> StrategyFidelityAssessment
        |
        +--> Causality validator/test contract --> CausalityAssessment
                                                   |
                                                   v
                                      ResearchAuthorization
```

The backtest runner may execute a package when policy permits mechanical validation. Risk and report consumers receive an explicit authorization result and must not infer authority from `provenance=passed` alone.

## Component 1: Strategy Contract

### Responsibility

Represent the authoritative trading requirements in a machine-readable artifact without claiming that free-form model output is authoritative.

### Proposed schema

Artifact type: `strategy.contract`
Schema version: `strategy-contract/v1`

```json
{
  "schema_version": "strategy-contract/v1",
  "contract_id": "strategy-contract-...",
  "strategy_source": {
    "evidence_ref": "...",
    "sha256": "...",
    "authority": "server_owned_extracted_source"
  },
  "execution_identity_hash": "...",
  "requirements": [
    {
      "requirement_id": "tf.structure.m15",
      "category": "timeframe_role",
      "required": true,
      "parameters": {"timeframe": "15m", "role": "structure"},
      "evidence_locator": "page:1-4"
    },
    {
      "requirement_id": "structure.body_close",
      "category": "signal_rule",
      "required": true,
      "parameters": {"applies_to": ["BOS", "CHoCH"]},
      "evidence_locator": "page:3"
    }
  ],
  "creation": {
    "method": "server_validated_extraction",
    "status": "verified"
  }
}
```

### Authority rule

The extracted source hash proves which text was used. It does not prove that an LLM-generated contract is correct. In v1, a contract becomes authoritative only through an explicit server-owned registration/validation step. If that validation is unavailable, `creation.status=unverified` and the final research authorization is downgraded.

Content-hash similarity may support deduplication and debugging. It must not establish semantic authority.

### Required requirement categories

- timeframe roles and allowed substitutions;
- VWAP price, volume, and reset/anchor rules;
- structure rules such as swing, BOS, CHoCH, body close, Order Block, and FVG;
- entry triggers and candle confirmation;
- exit, stop, target, and maximum-holding rules;
- position sizing and protective controls;
- session/time-zone assumptions;
- explicitly optional features.

## Component 2: Implementation Fidelity Assessment

### Responsibility

Bind the executed package to the strategy contract and expose every deviation structurally.

Artifact type: `strategy.fidelity_assessment`
Schema version: `strategy-fidelity/v1`

Each contract requirement receives exactly one implementation state:

```text
implemented_exact
implemented_equivalent
simplified
substituted
omitted
not_applicable
unknown
```

Each row contains:

```json
{
  "requirement_id": "tf.structure.m15",
  "state": "substituted",
  "implementation_ref": "code/signal_engine.py:69",
  "implementation_detail": {"actual_timeframe": "4h"},
  "authorized_deviation": false,
  "reason": "H4 structure replaced required M15 structure"
}
```

### Fidelity verdict

```text
passed
passed_with_authorized_simplifications
failed_required_rule
unknown_not_verified
```

Rules:

- any required `omitted`, unauthorized `substituted`, or `unknown` row prevents full-strategy research authorization;
- an explicitly declared baseline may execute with `simplified` rows but must be labeled as that baseline;
- the assessment is tied to the exact config and strategy SHA-256 hashes;
- changing either executable hash invalidates the assessment and requires regeneration;
- docstrings and worker prose may provide navigation hints but cannot satisfy a requirement.

## Component 3: Causal Higher-Timeframe Alignment

### Responsibility

Provide one reusable, testable boundary for attaching higher-timeframe information to lower-timeframe execution bars.

### Closed-bar invariant

At execution timestamp `t`, a higher-timeframe value is eligible only if its source bar closed no later than `t`.

For a left-labeled H1 bar covering `[10:00, 11:00)`, its high, low, close, and derived indicators must not be visible to an M5 decision at 10:05 through 10:55. They become eligible only after the bar is closed according to the timestamp convention declared by the input data.

### Interface behavior

The implementation should centralize the pattern rather than relying on individual strategy authors to remember `shift(1)`:

```text
resample completed HTF bars
-> compute HTF features
-> label each feature with availability timestamp
-> as-of join where available_at <= execution_timestamp
```

The exact Python API will be fixed in the implementation plan after repository inspection, but it must return both feature values and auditable availability metadata.

### Causality assessment

Artifact type: `strategy.causality_assessment`
Schema version: `strategy-causality/v1`

```json
{
  "status": "passed",
  "strategy_sha256": "...",
  "checks": {
    "prefix_invariance": "passed",
    "closed_bar_alignment": "passed",
    "same_bin_future_access": "passed",
    "execution_signal_lag": "passed"
  }
}
```

Statuses:

```text
passed
failed_prefix_invariance
failed_closed_bar_alignment
failed_execution_signal_lag
unknown_not_verified
```

## Component 4: Research Authorization

### Responsibility

Keep execution provenance, strategy fidelity, and causality independent, then determine the maximum allowed interpretation.

```json
{
  "execution_provenance": "passed",
  "strategy_fidelity": "failed_required_rule",
  "causality": "failed_prefix_invariance",
  "mechanical_backtest_completed": true,
  "research_authorization": "mechanical_validation_only",
  "allowed_conclusion": "NO_STRATEGY_VERDICT_AUTHORIZED"
}
```

Authorization states:

```text
full_strategy_research
declared_baseline_research
mechanical_validation_only
none
```

Minimum v1 rule:

```text
execution provenance passed
+ verified fidelity assessment acceptable for the declared target
+ causality passed
= eligible to proceed to later sufficiency/cost/risk gates
```

Passing this phase does not itself authorize an edge verdict. Later phases still control cost reconciliation, sample sufficiency, risk inference, and report language.

## Failure Semantics

Structured error codes:

```text
strategy_contract_unavailable
strategy_contract_unverified
strategy_fidelity_required_rule_missing
strategy_fidelity_unauthorized_substitution
strategy_fidelity_hash_mismatch
strategy_causality_prefix_invariance_failed
strategy_causality_closed_bar_failed
strategy_causality_execution_lag_failed
```

Behavior:

- validators never rewrite strategy code;
- validators never retry, fetch MT5 data, launch Swarm, or call a model;
- failure details name the requirement/check, expected value, observed value, and evidence reference;
- repeated validation is idempotent;
- a legacy artifact missing these fields becomes `unknown_not_verified` and does not crash;
- an ownership or hash mismatch must not mutate an existing assessment.

## Testing Strategy

### Level 1: Pure unit tests

- schema parsing and legacy defaults;
- exact/equivalent/simplified/substituted/omitted classification;
- executable-hash binding;
- closed-bar alignment for D1/H4/H1/M15 into M5;
- availability timestamp behavior;
- authorization-state calculation.

### Level 2: Metamorphic causality tests

The principal invariant is prefix stability:

```python
signals_prefix = generate(data.loc[:t])
signals_full = generate(data)
assert_series_equal(signals_prefix, signals_full.loc[:t])
```

Required perturbation cases:

- modify the final close of the current D1 bin;
- modify the future high/low of the current H1 bin;
- modify the future high/low of the current H4 bin;
- append future M5 bars;
- insert a later extreme that changes an unshifted pivot;
- verify no earlier signal changes.

### Level 3: Contract integration tests

- required M15 structure implemented as H4 without authorization -> fidelity failure;
- Order Block/FVG absent -> required-rule failure;
- explicitly declared simplified baseline -> mechanical execution allowed, full-PDF claim denied;
- strategy/config hash change -> stale assessment rejected;
- provenance PASS plus causality FAIL -> mechanical-only authorization;
- legacy run with no assessment -> unknown, no crash, no research upgrade.

### Level 4: Golden regression fixture

Create a bounded fixture derived from `swarm-20260908-231853-f91c2579` containing only the minimum authoritative data needed to reproduce:

- H1/H4/D1 same-bin look-ahead;
- PDF/implementation timeframe mismatch;
- missing M15/Order Block/FVG requirements;
- the fact that provenance PASS does not imply research authorization.

Do not copy the complete 500 KB equity file or full 5,074-bar snapshot into ordinary unit tests. Use a minimal synthetic bar sequence whose expected availability times are explicit. Synthetic data is acceptable for offline validator tests; it must be clearly marked and must never be promoted as research data.

### Level 5: Regression suite

The implementation plan must name the exact Python 3.13 test commands. It must include existing strict identity, strict bundle/preflight, artifact contract, window governance, report binding, and reconciliation suites affected by the new metadata.

No live Swarm, MT5, provider, or UI run is permitted during implementation. A controlled live acceptance is a later program gate.

## Cline Implementation Governance

### Roles

```text
Codex: spec owner, scope gate, test authority, diff reviewer, verifier, committer
Cline implementer: TDD implementation in a detached worktree, no commit
Cline reviewer: independent diff/test review, initially read-only
```

### Model routing

- `z-ai/glm-5.3-flash`: default for small, mechanical, tightly specified tasks;
- `deepseek/deepseek-v4-flash`: preferred for causality, state-machine, and cross-component tasks, or as an independent reviewer;
- paid/full variants are not assumed available;
- no model may approve its own work as the final gate.

### Execution rules

- one bounded task per detached worktree;
- main worktree is never edited by Cline;
- Cline does not commit;
- timeout 10-15 minutes per bounded task;
- at most 2-3 consecutive mistake retries;
- output is filtered to the final structured result;
- full reasoning/tool logs remain local and are read only when a failure requires root-cause evidence;
- Codex independently reads the touched diff and runs the required verification commands;
- only Codex stages the allowlisted files and commits after all gates pass.

### Required Cline result format

```text
TASK
DESIGN_DECISION
FILES_CHANGED
RED_TEST_EVIDENCE
TEST_COMMANDS
TEST_RESULTS
KNOWN_LIMITATIONS
SCOPE_CHECK
COMMIT_STATUS=NOT_COMMITTED
VERDICT
```

### Immediate rejection conditions

- modification outside the task allowlist;
- no demonstrated RED test;
- deletion or weakening of an existing test;
- live Swarm/MT5/provider/UI activity;
- hidden fallback or retry behavior;
- declaring PASS while acknowledging an untested affected consumer;
- changes made to the main worktree;
- attributing pre-existing failures to the patch without a clean-baseline comparison.

## Task Decomposition for the Future Implementation Plan

The implementation plan should split this design into independently reviewable tasks:

1. **Strategy-contract schemas and legacy parsing**
2. **Fidelity-assessment model and deterministic authorization rules**
3. **Minimal golden fidelity fixture from the known bad run**
4. **Causal HTF alignment primitive**
5. **Prefix-invariance and closed-bar test harness**
6. **Strict backtest/run-card integration**
7. **Research-authorization propagation and regression verification**

Each task must complete its own RED -> GREEN -> focused regression -> independent review cycle and produce a separate commit. Later tasks consume only explicitly named interfaces from earlier tasks.

## Acceptance Criteria

```text
STRATEGY CONTRACT VERSIONED AND HASH-BOUND: PASS
REQUIRED RULES HAVE EXPLICIT IMPLEMENTATION STATES: PASS
UNAUTHORIZED TIMEFRAME/RULE SUBSTITUTION DETECTED: PASS
FULL-PDF CLAIM BLOCKED WHEN FIDELITY FAILS: PASS
HTF FEATURES USE ONLY CLOSED BARS: PASS
PREFIX INVARIANCE ACROSS D1/H4/H1/M15: PASS
CAUSALITY ASSESSMENT HASH-BOUND: PASS
PROVENANCE, FIDELITY, AND CAUSALITY SEPARATED: PASS
MECHANICAL EXECUTION DOES NOT AUTHORIZE RESEARCH VERDICT: PASS
LEGACY ARTIFACTS DOWNGRADE WITHOUT CRASH: PASS
NO LIVE SIDE EFFECTS DURING IMPLEMENTATION: PASS
```

## Program Gate After Plan 1

When all acceptance criteria pass, the harness may say that a declared strategy or declared baseline was implemented causally. It still may not claim research validity or edge. The next program phase is `Cost Ledger + Metric Reconciliation`; live research reruns remain on hold until the later coverage, risk/report, and UI gates are also complete.
