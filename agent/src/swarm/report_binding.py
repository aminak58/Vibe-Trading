"""Deterministic final-strategy binding for strict Swarm reports.

This module intentionally does not interpret a worker's prose.  It builds a
small canonical strategy section from registered executed artifacts and lets
the runtime reject a competing free-form ``Final Strategy`` section.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from src.swarm.artifacts import current_artifact_manifest, verify_registered_artifact
from src.swarm.models import ArtifactRef


_REQUIRED_ARTIFACT_TYPES = frozenset(
    {
        "backtest.config",
        "backtest.run_card",
        "backtest.execution_provenance",
        "backtest.strategy",
    }
)
_FINAL_STRATEGY_HEADING = re.compile(r"(?im)^#{1,6}\s*final\s+strategy\b")


class NarrativeMismatch(ValueError):
    """Raised when a worker attempts to replace the executed strategy text."""


def _official_refs(
    *, run_dir: Path, refs: list[ArtifactRef], run_id: str, identity_hash: str
) -> dict[str, ArtifactRef]:
    selected: dict[str, ArtifactRef] = {}
    for ref in refs:
        if ref.artifact_type not in _REQUIRED_ARTIFACT_TYPES:
            continue
        if ref.execution_identity_hash != identity_hash:
            raise ValueError(f"executed artifact identity mismatch: {ref.artifact_type}")
        if (
            ref.producer_run_id != run_id
            or ref.producer_task_id != "task-backtest"
            or ref.provenance_status != "passed"
            or not verify_registered_artifact(run_dir, ref)
        ):
            raise ValueError(f"invalid executed artifact: {ref.artifact_type}")
        if ref.artifact_type in selected:
            raise ValueError(f"ambiguous executed artifact: {ref.artifact_type}")
        selected[ref.artifact_type] = ref
    missing = sorted(_REQUIRED_ARTIFACT_TYPES - set(selected))
    if missing:
        raise ValueError("missing executed artifacts: " + ", ".join(missing))
    manifest = current_artifact_manifest(run_dir, "task-backtest")
    if (
        manifest is None
        or not manifest.finalized
        or manifest.status != "succeeded"
        or manifest.run_id != run_id
        or manifest.producer_task_id != "task-backtest"
        or manifest.identity_hash != identity_hash
    ):
        raise ValueError("invalid executed artifact generation")
    for ref in selected.values():
        if (
            ref.manifest_generation_id != manifest.generation_id
            or ref.producer_attempt_id != manifest.producer_attempt_id
            or ref.artifact_id not in manifest.artifact_ids
        ):
            raise ValueError("stale executed artifact generation")
    return selected


def _read_json(run_dir: Path, ref: ArtifactRef) -> dict[str, Any]:
    try:
        value = json.loads((run_dir / ref.run_relative_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid executed artifact: {ref.artifact_type}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid executed artifact: {ref.artifact_type}")
    return value


def _strategy_docstring(run_dir: Path, ref: ArtifactRef) -> str:
    try:
        tree = ast.parse((run_dir / ref.run_relative_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ValueError("invalid executed artifact: backtest.strategy") from exc
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SignalEngine":
            docstring = ast.get_docstring(node, clean=True)
            if docstring:
                return docstring
    raise ValueError("executed strategy has no SignalEngine docstring")


def build_executed_strategy_binding(
    *, run_dir: Path, refs: list[ArtifactRef], run_id: str, identity_hash: str
) -> dict[str, str]:
    """Return canonical report fields from hash-verified executed artifacts."""
    official = _official_refs(
        run_dir=run_dir, refs=refs, run_id=run_id, identity_hash=identity_hash
    )
    config = _read_json(run_dir, official["backtest.config"])
    run_card = _read_json(run_dir, official["backtest.run_card"])
    provenance = _read_json(run_dir, official["backtest.execution_provenance"])
    if config.get("_execution_identity_hash") != identity_hash or provenance.get("identity_hash") != identity_hash:
        raise ValueError("executed artifact identity mismatch")
    reproducibility = run_card.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise ValueError("invalid executed artifact: backtest.run_card")
    if reproducibility.get("config_hash") != official["backtest.config"].sha256:
        raise ValueError("executed config hash mismatch")
    if reproducibility.get("strategy_hash") != official["backtest.strategy"].sha256:
        raise ValueError("executed strategy hash mismatch")
    codes = config.get("codes")
    if not isinstance(codes, list) or len(codes) != 1 or not isinstance(codes[0], str):
        raise ValueError("invalid executed artifact: backtest.config")
    source = config.get("source")
    interval = config.get("interval")
    if not isinstance(source, str) or not isinstance(interval, str):
        raise ValueError("invalid executed artifact: backtest.config")
    return {
        "symbol": codes[0],
        "source": source,
        "timeframe": interval,
        "identity_hash": identity_hash,
        "config_sha256": official["backtest.config"].sha256,
        "strategy_sha256": official["backtest.strategy"].sha256,
        "strategy_logic": _strategy_docstring(run_dir, official["backtest.strategy"]),
    }


def render_bound_strict_report(binding: dict[str, str], worker_report: str) -> str:
    """Prefix analysis with the immutable executed strategy section.

    A strict worker must not provide a second, free-form Final Strategy
    heading.  This is a structural check rather than similarity scoring.
    """
    if _FINAL_STRATEGY_HEADING.search(worker_report or ""):
        raise NarrativeMismatch("worker report attempted to replace Final Strategy")
    return (
        "## Final Strategy (Executed Contract)\n\n"
        f"- Resolved symbol: `{binding['symbol']}`\n"
        f"- Source: `{binding['source']}`\n"
        f"- Execution timeframe: `{binding['timeframe']}`\n"
        f"- Execution identity hash: `{binding['identity_hash']}`\n"
        f"- Config SHA-256: `{binding['config_sha256']}`\n"
        f"- Strategy SHA-256: `{binding['strategy_sha256']}`\n\n"
        "### Executed Strategy Logic\n\n"
        + binding["strategy_logic"]
        + "\n\n## Research Report\n\n"
        + (worker_report or "")
    )
