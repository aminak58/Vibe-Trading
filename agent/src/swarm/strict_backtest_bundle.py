"""Server-owned strict backtest worker input and package preflight."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from src.execution_identity import (
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    SourceMode,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STRATEGY_SOURCE_USER_VAR = "__strict_strategy_source_v1"


@dataclass(frozen=True)
class StrategySource:
    """Server-issued availability record for a required strategy source."""

    status: Literal["available", "unavailable"]
    evidence_ref: str | None = None
    content_hash: str | None = None
    materialized_path: str | None = None


@dataclass(frozen=True)
class StrictBacktestInputBundle:
    """Immutable strict backtest package contract supplied by the runtime."""

    requested_symbol: str
    resolved_symbol: str
    codes: tuple[str, ...]
    source: str
    source_mode: str
    fallback: str
    synthetic: str
    expected_config_path: str
    expected_strategy_path: str
    window_authority: dict[str, Any]
    strategy_source: StrategySource


def materialize_strategy_source(
    artifact_dir: Path,
    *,
    text: str,
    evidence_ref: str,
) -> StrategySource:
    """Persist server-extracted strategy text for one worker artifact root."""
    if not isinstance(text, str) or not text.strip() or not isinstance(evidence_ref, str) or not evidence_ref:
        return StrategySource(status="unavailable")
    path = artifact_dir / "strategy_source.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return StrategySource(
        status="available",
        evidence_ref=evidence_ref,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        materialized_path=path.name,
    )


def strategy_source_payload(text: str, evidence_ref: str) -> str:
    """Encode server-extracted strategy text for trusted Swarm transport."""
    if not isinstance(text, str) or not text.strip() or not isinstance(evidence_ref, str) or not evidence_ref:
        raise ValueError("strategy source payload requires non-empty extracted text and evidence ref")
    return json.dumps(
        {
            "schema_version": "strict-strategy-source/v1",
            "text": text,
            "evidence_ref": evidence_ref,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def strategy_source_from_user_vars(user_vars: dict[str, str], artifact_dir: Path) -> StrategySource:
    """Materialize only a server-issued extracted-text payload for a worker."""
    raw = user_vars.get(STRATEGY_SOURCE_USER_VAR)
    if not isinstance(raw, str):
        return StrategySource(status="unavailable")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return StrategySource(status="unavailable")
    if not isinstance(payload, dict) or payload.get("schema_version") != "strict-strategy-source/v1":
        return StrategySource(status="unavailable")
    text = payload.get("text")
    evidence_ref = payload.get("evidence_ref")
    declared_hash = payload.get("content_hash")
    if (
        not isinstance(text, str)
        or not isinstance(evidence_ref, str)
        or not isinstance(declared_hash, str)
        or hashlib.sha256(text.encode("utf-8")).hexdigest() != declared_hash
    ):
        return StrategySource(status="unavailable")
    return materialize_strategy_source(artifact_dir, text=text, evidence_ref=evidence_ref)


def is_strict_backtest_identity(identity: ExecutionIdentity | None) -> bool:
    """Return whether identity needs the strict backtest package contract."""
    return bool(
        identity
        and identity.status is ExecutionIdentityStatus.VERIFIED
        and identity.mode is ExecutionMode.SOURCE_SCOPED
        and identity.policy.source_mode is SourceMode.STRICT
    )


def build_strict_backtest_input_bundle(
    identity: ExecutionIdentity,
    *,
    window_authority: dict[str, Any] | None,
    strategy_source: StrategySource,
) -> StrictBacktestInputBundle:
    """Build a strict package contract from the verified server identity."""
    if not is_strict_backtest_identity(identity):
        raise ValueError("strict backtest input bundle requires a verified strict execution identity")
    if len(identity.requests) != 1 or len(identity.resolutions) != 1:
        raise ValueError("strict backtest input bundle requires exactly one request and resolution")
    request = identity.requests[0]
    resolution = identity.resolutions[0]
    if (
        not request.symbol
        or not request.source
        or not resolution.resolved_symbol
        or resolution.request_id != request.request_id
        or (resolution.source or "").casefold() != request.source.casefold()
    ):
        raise ValueError("strict backtest input bundle requires a same-source verified resolution")

    return StrictBacktestInputBundle(
        requested_symbol=request.symbol,
        resolved_symbol=resolution.resolved_symbol,
        codes=(resolution.resolved_symbol,),
        source=request.source,
        source_mode=identity.policy.source_mode.value,
        fallback=identity.policy.fallback.value,
        synthetic=identity.policy.synthetic.value,
        expected_config_path="config.json",
        expected_strategy_path="code/signal_engine.py",
        window_authority=dict(window_authority or {}),
        strategy_source=strategy_source,
    )


def _error(error_code: str, message: str) -> dict[str, str]:
    return {"status": "error", "error_code": error_code, "error": message}


def _strategy_source_is_valid(source: StrategySource) -> bool:
    return bool(
        source.status == "available"
        and source.evidence_ref
        and source.materialized_path
        and source.content_hash
        and _SHA256_RE.fullmatch(source.content_hash)
    )


def _valid_window(config: dict[str, Any]) -> bool:
    start = config.get("start_date")
    end = config.get("end_date")
    if not isinstance(start, str) or not isinstance(end, str) or not start or not end:
        return False
    try:
        return date.fromisoformat(start) <= date.fromisoformat(end)
    except ValueError:
        return False


def validate_strict_backtest_package(
    run_dir: Path,
    bundle: StrictBacktestInputBundle,
) -> dict[str, str] | None:
    """Validate a strict worker package before backtest acquisition may start."""
    if not _strategy_source_is_valid(bundle.strategy_source):
        return _error(
            "required_strategy_source_unavailable",
            "A required authoritative strategy source is unavailable to the strict backtest worker.",
        )

    config_path = run_dir / bundle.expected_config_path
    if not config_path.is_file():
        return _error("invalid_backtest_package_path", "config.json not found at the canonical package path.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _error("invalid_backtest_config", "config.json is not valid UTF-8 JSON.")
    if not isinstance(config, dict):
        return _error("invalid_backtest_config", "config.json must contain a JSON object.")
    if config.get("codes") != list(bundle.codes) or config.get("source") != bundle.source:
        return _error(
            "invalid_backtest_config_identity",
            "config.json codes/source do not match the verified strict backtest input bundle.",
        )
    if not _valid_window(config):
        return _error("invalid_window_config", "config.json requires an ordered ISO start_date and end_date.")

    strategy_path = run_dir / bundle.expected_strategy_path
    if not strategy_path.is_file():
        return _error(
            "invalid_backtest_package_path",
            "code/signal_engine.py not found at the canonical strict package path.",
        )
    return None
