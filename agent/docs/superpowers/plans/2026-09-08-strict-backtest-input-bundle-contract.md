# Strict Backtest Input Bundle Contract v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent strict Swarm backtest workers from guessing identity fields, package paths, window fields, or strategy-source availability before `backtest()`.

**Architecture:** A new pure bundle/preflight module derives immutable strict fields from `ExecutionIdentity`. `run_worker` injects the bundle into strict task-backtest context and validates the package before delegating a backtest tool call. `BacktestTool` remains unchanged as the MT5 acquisition boundary.

**Tech Stack:** Python 3.13, pytest, Pydantic execution identity models, existing Swarm worker/runtime and BacktestTool.

**Spec:** `docs/superpowers/specs/2026-09-08-strict-backtest-input-bundle-contract-design.md`

## Global Constraints

- No live Swarm, MT5 fetch, provider/model call, UI E2E, or live backtest in tests.
- Do not modify MT5 snapshots/loaders, artifact promotion, risk/report, parent orchestration, or retry/fallback behavior.
- Derive `codes` only from `ExecutionIdentity`; model fields do not establish identity.
- Strict paths are `config.json` and `code/signal_engine.py`.
- An unavailable required strategy source fails before code generation/backtest.
- Use `py -3.13 -m pytest`.

---

### Task 1: Bundle and pure package preflight

**Files:**
- Create: `src/swarm/strict_backtest_bundle.py`
- Create: `tests/test_strict_backtest_input_bundle.py`

**Interfaces:**
- Produces `StrictBacktestInputBundle`, `build_strict_backtest_input_bundle(identity, *, window_authority, strategy_source)`, and `validate_strict_backtest_package(run_dir, bundle)`.
- `validate_strict_backtest_package` returns `None` or a structured `error_code` without loading MT5.

- [ ] **Step 1: Write failing tests**

```python
def test_verified_mt5_identity_builds_canonical_codes_bundle():
    bundle = build_strict_backtest_input_bundle(identity, window_authority={}, strategy_source=available_source)
    assert bundle.codes == ("XAUUSD_o",)
    assert bundle.expected_config_path == "config.json"
    assert bundle.expected_strategy_path == "code/signal_engine.py"

def test_config_without_codes_fails_before_acquisition(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"source": "mt5", "symbol": "XAUUSD_o"}))
    assert validate_strict_backtest_package(tmp_path, bundle)["error_code"] == "invalid_backtest_config_identity"
```

- [ ] **Step 2: Verify RED**

Run: `py -3.13 -m pytest -q tests/test_strict_backtest_input_bundle.py`

Expected: FAIL because the module and functions do not exist.

- [ ] **Step 3: Implement minimally**

Create a frozen bundle with requested/resolved symbols, canonical `codes`, strict policy, canonical paths, window metadata, and strategy-source availability/ref. Validate document availability, JSON/object validity, exact `codes`, `source`, dates, and path. Return exactly one of `required_strategy_source_unavailable`, `invalid_backtest_config_identity`, `invalid_window_config`, or `invalid_backtest_package_path`.

- [ ] **Step 4: Verify GREEN**

Run: `py -3.13 -m pytest -q tests/test_strict_backtest_input_bundle.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/swarm/strict_backtest_bundle.py tests/test_strict_backtest_input_bundle.py`

Run: `git commit -m "feat: add strict backtest input bundle preflight"`

### Task 2: Worker injection and one-shot enforcement

**Files:**
- Modify: `src/swarm/worker.py` prompt construction and `backtest` tool-call branch.
- Modify: `tests/test_strict_backtest_input_bundle.py`.

**Interfaces:**
- Consumes the Task 1 bundle and `run_worker` task context.
- Produces a server-rendered bundle notice and a local structured result without calling the tool when preflight fails.

- [ ] **Step 1: Write failing tests**

```python
def test_wrong_strategy_path_never_delegates_backtest(monkeypatch, tmp_path):
    # Root signal_engine.py is invalid for a strict worker.
    assert observed_backtest_calls == []
    assert "invalid_backtest_package_path" in worker_tool_results

def test_second_blind_backtest_is_suppressed_after_preflight_failure(monkeypatch, tmp_path):
    assert observed_backtest_calls == []
    assert worker_tool_results.count("invalid_backtest_config_identity") == 1
```

- [ ] **Step 2: Verify RED**

Run: `py -3.13 -m pytest -q tests/test_strict_backtest_input_bundle.py -k "worker or blind"`

Expected: FAIL because the worker delegates invalid `backtest({})` calls today.

- [ ] **Step 3: Implement minimally**

After `artifact_dir` resolution, build one bundle for a strict task-backtest and add its server-generated content to the worker context. Before registry execution of `backtest`, run Task 1 preflight. Cache the first preflight failure and suppress subsequent blind calls for that task.

- [ ] **Step 4: Verify GREEN**

Run: `py -3.13 -m pytest -q tests/test_strict_backtest_input_bundle.py -k "worker or blind"`

Expected: PASS with zero mocked backtest executions for invalid packages.

- [ ] **Step 5: Commit**

Run: `git add src/swarm/worker.py tests/test_strict_backtest_input_bundle.py`

Run: `git commit -m "feat: preflight strict backtest worker packages"`

### Task 3: Acquisition reachability and regression suite

**Files:**
- Modify: `tests/test_strict_backtest_input_bundle.py`
- Modify: `tests/test_backtest_preflight_barrier.py` only when a compatibility assertion is necessary.

**Interfaces:**
- Consumes valid Task 1/2 bundle/package.
- Proves valid strict input reaches mocked `prepare_mt5_snapshot`; non-strict workers bypass the bundle path.

- [ ] **Step 1: Write failing tests**

```python
def test_valid_strict_package_reaches_mocked_snapshot(monkeypatch, tmp_path):
    write_valid_config(tmp_path, codes=["XAUUSD_o"])
    (tmp_path / "code" / "signal_engine.py").write_text("class SignalEngine: pass\\n")
    assert validate_strict_backtest_package(tmp_path, bundle) is None
    assert mocked_snapshot_calls == [tmp_path]

def test_non_strict_worker_bypasses_strict_bundle():
    assert build_bundle_for_worker(non_strict_identity, window_authority={}, strategy_source=None) is None
```

- [ ] **Step 2: Verify RED**

Run: `py -3.13 -m pytest -q tests/test_strict_backtest_input_bundle.py -k "snapshot or non_strict"`

Expected: FAIL until valid-package worker wiring exists.

- [ ] **Step 3: Implement minimally**

Pass valid strict packages through preflight without changing BacktestTool acquisition. Return no bundle/preflight behavior for non-strict workers.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `py -3.13 -m pytest -q tests/test_strict_backtest_input_bundle.py tests/test_backtest_preflight_barrier.py tests/test_mt5_snapshot_handoff.py tests/test_strict_backtest_task_output_contract.py tests/test_swarm_execution_identity.py tests/test_parent_pre_dispatch_obligation_guard.py`

Expected: all pass with no live external action.

- [ ] **Step 5: Commit**

Run: `git add tests/test_strict_backtest_input_bundle.py tests/test_backtest_preflight_barrier.py`

Run: `git commit -m "test: cover strict backtest bundle handoff"`

## Plan self-review

- Task 1 implements identity/path/document/window validation.
- Task 2 makes runtime authoritative and prevents repeated blind calls.
- Task 3 proves the valid acquisition boundary and non-strict compatibility.
- The plan does not change excluded MT5, artifact, report, risk, parent, or retry behavior.
