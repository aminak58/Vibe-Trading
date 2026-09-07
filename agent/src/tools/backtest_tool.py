"""Backtest execution tool: validates config.json + signal_engine.py and runs the built-in engine."""

from __future__ import annotations

import json
from pathlib import Path

from backtest.loaders.registry import VALID_SOURCES
from src.agent.progress import emit_progress
from src.agent.tools import BaseTool
from src.core.runner import Runner
from src.execution_identity import ExecutionIdentity, ExecutionIdentityStatus, ExecutionMode, SourceMode
from src.tools.path_utils import safe_run_dir


def _persist_config(path: Path, config: dict) -> None:
    """Atomically persist parent-added execution metadata for the child."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _apply_strict_execution_identity(config: dict, identity: ExecutionIdentity | None) -> str | None:
    """Freeze a verified server-issued identity into a generated backtest config.

    ``backtest`` accepts only a ``run_dir`` publicly, so this is the final
    server boundary that prevents generated config.json from escaping an
    already-verified source-scoped execution contract.
    """
    if (
        identity is None
        or identity.mode is not ExecutionMode.SOURCE_SCOPED
        or identity.policy.source_mode is not SourceMode.STRICT
    ):
        return None
    if identity.status is not ExecutionIdentityStatus.VERIFIED or not identity.requests:
        return "Strict execution identity is not verified."
    request = identity.requests[0]
    resolutions = [item for item in identity.resolutions if item.request_id == request.request_id]
    if len(resolutions) != 1 or not resolutions[0].resolved_symbol or not request.source:
        return "Strict execution identity has no verified source-native resolution."
    expected_symbols = {resolutions[0].resolved_symbol.upper()}
    configured_symbols = {str(item).upper() for item in (config.get("codes") or [])}
    if configured_symbols != expected_symbols:
        return "Backtest config symbol does not match the verified resolved execution identity."
    if str(config.get("source") or "").casefold() != request.source.casefold():
        return "Backtest config source does not match the strict execution identity."
    config["_execution_policy"] = identity.policy.model_dump(mode="json")
    config["_execution_identity_hash"] = identity.identity_hash
    config["_execution_identity"] = {
        "requested_symbol": request.symbol,
        "resolved_symbol": resolutions[0].resolved_symbol,
        "source": request.source,
        "market": resolutions[0].market,
        "asset_class": resolutions[0].asset_class,
    }
    return None


def _persist_strict_execution_provenance(
    run_path: Path,
    config: dict,
    *,
    owning_run_id: str | None,
) -> None:
    """Persist structured strict provenance; never derive it from worker prose."""
    identity = config.get("_execution_identity")
    if not isinstance(identity, dict) or not config.get("_execution_identity_hash"):
        return
    payload = {
        "identity_hash": config["_execution_identity_hash"],
        "owning_run_id": owning_run_id,
        "requested_symbol": identity.get("requested_symbol"),
        "resolved_symbol": identity.get("resolved_symbol"),
        "effective_source": config.get("source"),
        "market": identity.get("market"),
        "asset_class": identity.get("asset_class"),
        "synthetic": False,
        "fallback_used": False,
    }
    (run_path / "execution_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_backtest(
    run_dir: str,
    *,
    execution_identity: ExecutionIdentity | None = None,
    owning_run_id: str | None = None,
    window_authority: dict | None = None,
) -> str:
    """Run backtest: validate config.json + signal_engine.py, invoke built-in engine.

    Args:
        run_dir: Path to the run directory.

    Returns:
        JSON-formatted execution result.
    """
    emit_progress("validate", message="validating run_dir and config")
    try:
        run_path = safe_run_dir(run_dir)
    except ValueError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)

    config_path = run_path / "config.json"
    if not config_path.exists():
        return json.dumps({"status": "error", "error": "config.json not found"}, ensure_ascii=False)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"config.json parse error: {e}"}, ensure_ascii=False)

    if not isinstance(config, dict):
        return json.dumps({"status": "error", "error": "config.json must contain an object"}, ensure_ascii=False)

    if "source" not in config:
        return json.dumps({"status": "error", "error": "config.json missing 'source' field (tushare/okx/yfinance)"}, ensure_ascii=False)

    identity_error = _apply_strict_execution_identity(config, execution_identity)
    if identity_error is not None:
        return json.dumps(
            {"status": "error", "error_code": "identity_blocked", "error": identity_error},
            ensure_ascii=False,
        )

    if config["source"] not in VALID_SOURCES:
        return json.dumps({"status": "error", "error": f"source must be one of {VALID_SOURCES}, got: {config['source']}"}, ensure_ascii=False)

    # Package preflight is structural only. Never import generated strategy
    # code in the trusted parent, and do not freeze data for a missing package.
    signal_path = run_path / "code" / "signal_engine.py"
    if not signal_path.is_file():
        return json.dumps({"status": "error", "error": "code/signal_engine.py not found or not a regular file"}, ensure_ascii=False)

    # MT5's terminal IPC is process/environment-sensitive. Acquire broker bars
    # before crossing into the generated-strategy sandbox, then make the child
    # consume only an immutable verified snapshot. Explicit MT5 requests are
    # intentionally fail-closed: no loader fallback is allowed here or below.
    if config["source"] == "mt5":
        try:
            from backtest.mt5_snapshot import MANIFEST_RELATIVE_PATH, prepare_mt5_snapshot

            manifest = prepare_mt5_snapshot(run_path, config)
            config["mt5_snapshot_manifest"] = MANIFEST_RELATIVE_PATH.as_posix()
            config["mt5_snapshot_sha256"] = manifest["sha256"]
            # The child receives only this parent-resolved immutable contract;
            # it must not look up broker costs or retain request-time defaults.
            config["cost_model"] = manifest["cost_model"]
            _persist_config(config_path, config)
        except Exception as exc:  # noqa: BLE001 - surface one strict handoff envelope
            return json.dumps(
                {"status": "error", "error": f"MT5-backed data acquisition/handoff failure: {exc}"},
                ensure_ascii=False,
            )

    if execution_identity is not None:
        try:
            _persist_config(config_path, config)
            _persist_strict_execution_provenance(
                run_path,
                config,
                owning_run_id=owning_run_id,
            )
        except OSError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error_code": "identity_blocked",
                    "error": f"Could not persist strict execution provenance: {exc}",
                },
                ensure_ascii=False,
            )

    agent_root = Path(__file__).resolve().parents[2]
    entry_script = agent_root / "backtest" / "runner.py"

    source = config.get("source", "?")
    emit_progress(
        "simulate",
        message=f"running backtest engine (source={source})",
    )
    runner = Runner(timeout=300)
    result = runner.execute(
        entry_script,
        run_path,
        cwd=agent_root,
        cli_args=[str(run_path)],
    )

    # The engine writes run_card.json inside the generated-strategy process.
    # Add research-window authority only after that process ends, using an
    # internal value supplied by the trusted parent.  Config content or a
    # matching config hash is never treated as authority.
    if window_authority is not None:
        try:
            from backtest.research_window import (
                finalize_research_window_metadata,
                persist_server_window_authority,
            )

            persist_server_window_authority(run_path, window_authority)
            finalize_research_window_metadata(run_path, window_authority)
        except OSError:
            # The engine's legacy card remains readable and consequently has
            # unknown authority; do not turn a mechanical execution into a
            # false research conclusion because a post-run audit write failed.
            pass

    emit_progress("finalize", message="collecting artifacts")
    artifacts_found = {name: str(path) for name, path in result.artifacts.items()}
    response = {
        "status": "ok" if result.success else "error",
        "exit_code": result.exit_code,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        "artifacts": artifacts_found,
        "run_dir": run_dir,
    }
    if config.get("source") == "mt5" and isinstance(config.get("cost_model"), dict):
        response["cost_grounding"] = str(config["cost_model"].get("mode", "")).upper()
        response["cost_model"] = config["cost_model"]
    return json.dumps(response, ensure_ascii=False)


class BacktestTool(BaseTool):
    """Backtest execution tool."""

    name = "backtest"
    description = "Run backtest: validate config.json + signal_engine.py, invoke built-in engine."
    parameters = {
        "type": "object",
        "properties": {
            "run_dir": {"type": "string", "description": "Path to the run directory"},
        },
        "required": ["run_dir"],
    }
    repeatable = True
    is_readonly = False

    def execute(self, **kwargs) -> str:
        """Execute backtest."""
        return run_backtest(
            kwargs["run_dir"],
            execution_identity=kwargs.get("__execution_identity"),
            owning_run_id=kwargs.get("__swarm_run_id"),
            window_authority=kwargs.get("__window_authority"),
        )
