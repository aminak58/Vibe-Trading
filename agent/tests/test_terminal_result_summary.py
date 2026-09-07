"""Canonical terminal-result summaries for all parent recovery paths."""

from __future__ import annotations

from datetime import datetime, timezone

from src.swarm.models import ArtifactRef, RunStatus, SwarmRun, SwarmTask
from src.agent.terminal_summary import summarize_terminal_run


def _ref(kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact-{kind}",
        producer_run_id="run-1",
        producer_task_id="task-backtest",
        producer_agent_id="backtester",
        run_relative_path=f"artifacts/backtester/{kind}.json",
        sha256="a" * 64,
        byte_size=1,
        execution_identity_hash="identity-1",
        artifact_type=kind,
        provenance_status="passed",
    )


def test_failed_summary_prefers_executable_root_failure_over_blocked_report() -> None:
    """Reordering tasks must not turn a downstream block into the root cause."""
    run = SwarmRun(
        id="run-1",
        preset_name="quant_scalp_desk",
        status=RunStatus.failed,
        created_at=datetime.now(timezone.utc).isoformat(),
        identity_hash="identity-1",
        provenance_validation_status="passed",
        tasks=[
            SwarmTask(
                id="task-report", agent_id="report_aggregator", prompt_template="x",
                status="blocked", error="Blocked: upstream not completed (task-risk=failed)",
            ),
            SwarmTask(
                id="task-risk", agent_id="risk_auditor", prompt_template="x",
                status="failed", error="provider_stream_error: 429 INFERENCE_CAP_ERROR",
            ),
            SwarmTask(
                id="task-backtest", agent_id="backtester", prompt_template="x",
                status="completed", artifact_refs=[
                    _ref(kind) for kind in (
                        "backtest.execution_provenance", "backtest.config", "backtest.run_card",
                        "backtest.strategy", "backtest.metrics", "backtest.trades", "backtest.equity",
                    )
                ],
            ),
        ],
    )

    summary = summarize_terminal_run(run)

    assert summary["terminal_reason"] == "provider_stream_error: 429 INFERENCE_CAP_ERROR"
    assert summary["root_failure"] == {
        "task_id": "task-risk",
        "agent_id": "risk_auditor",
        "error": "provider_stream_error: 429 INFERENCE_CAP_ERROR",
    }
    assert summary["downstream_consequences"] == [{
        "task_id": "task-report",
        "agent_id": "report_aggregator",
        "status": "blocked",
        "error": "Blocked: upstream not completed (task-risk=failed)",
    }]
    assert summary["partial_backtest"] == {
        "completed": True,
        "provenance_validation_status": "passed",
        "official_artifact_types": [
            "backtest.config", "backtest.equity", "backtest.execution_provenance",
            "backtest.metrics", "backtest.run_card", "backtest.strategy", "backtest.trades",
        ],
        "research_pipeline_incomplete": True,
    }


def test_missing_or_unprovenanced_backtest_refs_never_claim_official_partial_backtest() -> None:
    run = SwarmRun(
        id="run-1", preset_name="quant_scalp_desk", status=RunStatus.failed,
        created_at=datetime.now(timezone.utc).isoformat(), identity_hash="identity-1",
        provenance_validation_status="passed",
        tasks=[
            SwarmTask(id="task-backtest", agent_id="backtester", prompt_template="x", status="completed"),
            SwarmTask(id="task-risk", agent_id="risk_auditor", prompt_template="x", status="failed", error="risk failed"),
        ],
    )

    summary = summarize_terminal_run(run)

    assert summary["partial_backtest"]["completed"] is False
    assert summary["partial_backtest"]["official_artifact_types"] == []
