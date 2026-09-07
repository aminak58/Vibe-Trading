"""Server-owned research-window metadata for backtest run cards.

Window authority is deliberately independent of source provenance.  A config
file can be copied verbatim by an untrusted worker, so its contents or hash
can never establish who selected its dates.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping


_AUTHORITY_SOURCES = frozenset(
    {"current_user_explicit", "attachment_explicit", "preset_default", "prepared_config_reuse", "worker_generated"}
)
_AUTHORITY_SIDECAR = ".server_research_window_authority.json"


def persist_server_window_authority(run_dir: Path, authority: Mapping[str, Any]) -> None:
    """Persist server-issued authority only after untrusted execution ends.

    The sidecar is deliberately written by the trusted backtest-tool parent,
    not read by the generated strategy process.  It is audit evidence for the
    authority passed to the post-execution run-card finalizer; it is never
    inferred from config contents or hashes.
    """
    path = run_dir / _AUTHORITY_SIDECAR
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(authority), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_server_window_authority(run_dir: Path) -> dict[str, Any] | None:
    """Load server-written authority sidecar conservatively for audit use."""
    path = run_dir / _AUTHORITY_SIDECAR
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def finalize_research_window_metadata(run_dir: Path, authority: Mapping[str, Any]) -> dict[str, Any] | None:
    """Attach trusted window metadata to an already-written canonical run card.

    Backtest engines write their run card inside the generated-strategy
    subprocess.  The trusted parent performs this finalization *after* that
    subprocess exits, so a worker-writable config cannot self-assert authority.
    """
    run_card_path = run_dir / "run_card.json"
    config_path = run_dir / "config.json"
    try:
        card = json.loads(run_card_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(card, dict) or not isinstance(config, Mapping):
        return None
    metrics = card.get("metrics")
    metadata = build_research_window_metadata(
        config,
        metrics if isinstance(metrics, Mapping) else {},
        authority=authority,
    )
    card["research_window"] = metadata
    temporary = run_card_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_card_path)
    return metadata


def build_research_window_metadata(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build conservative, serializable metadata for a completed backtest.

    ``authority`` is an internal, server-injected value.  Missing or
    untrusted authority is intentionally represented as ``unknown`` rather
    than inferred from config contents, hashes, or worker prose.
    """
    window_authority = _normalize_authority(authority)
    snapshot = config.get("_mt5_snapshot_provenance")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    validation = metrics.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    monte_carlo = validation.get("monte_carlo")
    monte_carlo = monte_carlo if isinstance(monte_carlo, Mapping) else {}
    cost_model = config.get("cost_model")
    cost_model = cost_model if isinstance(cost_model, Mapping) else {}

    return {
        "window_authority": window_authority,
        "coverage_probe": {"status": "not_performed"},
        "history_depth_claim_supported": False,
        "sample": {
            "bars_count": _int_or_none(snapshot.get("row_count")),
            "date_span_days": _date_span(config.get("start_date"), config.get("end_date")),
            "trading_days_count": None,
            "trades_count": _int_or_none(metrics.get("trade_count")),
            "oos_present": False,
            "monte_carlo_runs": _int_or_none(monte_carlo.get("n_simulations")),
            "cost_model_quality": _cost_model_quality(cost_model),
        },
        "research_sufficiency": {"status": "unknown_not_enforced"},
        "official_conclusion": "EDGE NOT SHOWN ON LIMITED BASELINE",
    }


def _normalize_authority(authority: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(authority, Mapping) or authority.get("server_owned") is not True:
        return {"source": "unknown", "evidence_ref": None, "user_explicit": False}
    source = authority.get("source")
    if source not in _AUTHORITY_SOURCES:
        return {"source": "unknown", "evidence_ref": None, "user_explicit": False}

    normalized: dict[str, Any] = {
        "source": source,
        "evidence_ref": _string_or_none(authority.get("evidence_ref")),
        "user_explicit": bool(authority.get("user_explicit", False)),
    }
    if source == "prepared_config_reuse":
        normalized.update(
            {
                "source_run_id": _string_or_none(authority.get("source_run_id")),
                "current_run_id": _string_or_none(authority.get("current_run_id")),
                "reuse_approved": bool(authority.get("reuse_approved", False)),
                "containment_valid": bool(authority.get("containment_valid", False)),
            }
        )
    return normalized


def _date_span(start: Any, end: Any) -> int | None:
    try:
        return (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days + 1
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _cost_model_quality(cost_model: Mapping[str, Any]) -> str:
    mode = cost_model.get("mode")
    if mode == "user_assumption":
        return "user_assumption"
    if mode:
        return "broker_grounded"
    return "unknown"
