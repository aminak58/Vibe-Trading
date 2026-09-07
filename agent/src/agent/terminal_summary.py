"""Canonical, server-owned summaries for terminal Swarm recovery paths."""

from __future__ import annotations

from typing import Any


_STRICT_BACKTEST_TYPES = {
    "backtest.execution_provenance",
    "backtest.config",
    "backtest.run_card",
    "backtest.strategy",
    "backtest.metrics",
    "backtest.trades",
    "backtest.equity",
}


def _status(value: object) -> str:
    return str(getattr(value, "value", value) or "").casefold()


def summarize_terminal_run(run: Any) -> dict[str, Any]:
    """Return one provenance-backed terminal summary for all recovery paths.

    Failed executable tasks are root causes; blocked tasks are only their
    downstream consequences.  Worker prose never participates.
    """
    tasks = list(getattr(run, "tasks", []) or [])
    failed = [task for task in tasks if _status(getattr(task, "status", "")) == "failed"]
    blocked = [task for task in tasks if _status(getattr(task, "status", "")) == "blocked"]
    root = failed[0] if failed else None
    root_error = str(getattr(root, "error", "") or "").strip() if root is not None else ""
    root_failure = (
        {
            "task_id": str(getattr(root, "id", "")),
            "agent_id": str(getattr(root, "agent_id", "")),
            "error": root_error or "owned_swarm_terminal_failure",
        }
        if root is not None
        else None
    )

    run_id = getattr(run, "id", None)
    identity_hash = getattr(run, "identity_hash", None)
    official_types = sorted(
        {
            str(getattr(ref, "artifact_type", ""))
            for task in tasks
            for ref in getattr(task, "artifact_refs", [])
            if str(getattr(ref, "artifact_type", "")).startswith("backtest.")
            and getattr(ref, "producer_run_id", None) == run_id
            and getattr(ref, "execution_identity_hash", None) == identity_hash
            and getattr(ref, "provenance_status", None) == "passed"
        }
    )
    backtest_completed = any(
        str(getattr(task, "id", "")) == "task-backtest"
        and _status(getattr(task, "status", "")) == "completed"
        for task in tasks
    )
    provenance = _status(getattr(run, "provenance_validation_status", ""))
    official_completed = (
        backtest_completed
        and provenance == "passed"
        and _STRICT_BACKTEST_TYPES.issubset(official_types)
    )

    return {
        "terminal_reason": root_failure["error"] if root_failure else "owned_swarm_terminal_failure",
        "root_failure": root_failure,
        "downstream_consequences": [
            {
                "task_id": str(getattr(task, "id", "")),
                "agent_id": str(getattr(task, "agent_id", "")),
                "status": _status(getattr(task, "status", "")),
                "error": str(getattr(task, "error", "") or "").strip(),
            }
            for task in blocked
        ],
        "artifact_status": {
            str(getattr(task, "id", "")): {
                "status": _status(getattr(task, "status", "")),
                "artifact_count": len(getattr(task, "artifact_refs", [])),
            }
            for task in tasks
        },
        "official_backtest_artifact_types": official_types,
        "partial_backtest": {
            "completed": official_completed,
            "provenance_validation_status": provenance or "not_required",
            "official_artifact_types": official_types,
            "research_pipeline_incomplete": bool(failed or blocked),
        },
    }
