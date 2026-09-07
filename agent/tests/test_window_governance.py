"""Strict research-window governance must remain separate from provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace

from backtest.research_window import (
    build_research_window_metadata,
    finalize_research_window_metadata,
    load_server_window_authority,
    persist_server_window_authority,
)
from backtest.run_card import write_run_card
from src.agent.current_user_intent import window_authority_from_current_user_message
from src.swarm.report_binding import NarrativeMismatch, render_bound_strict_report
from src.tools import backtest_tool


def _config() -> dict:
    return {
        "source": "mt5",
        "codes": ["XAUUSD_o"],
        "start_date": "2026-08-20",
        "end_date": "2026-09-04",
        "interval": "5m",
        "cost_model": {"mode": "user_assumption"},
        "_mt5_snapshot_provenance": {"row_count": 3310},
    }


def test_current_user_explicit_dates_are_recorded_as_window_authority() -> None:
    metadata = build_research_window_metadata(
        _config(),
        {"trade_count": 31, "validation": {"monte_carlo": {"n_simulations": 5}}},
        authority={
            "source": "current_user_explicit",
            "evidence_ref": "request:current#dates",
            "user_explicit": True,
            "server_owned": True,
        },
    )

    assert metadata["window_authority"]["source"] == "current_user_explicit"
    assert metadata["window_authority"]["user_explicit"] is True
    assert metadata["window_authority"]["evidence_ref"] == "request:current#dates"


def test_current_user_date_parser_uses_only_raw_message_dates() -> None:
    authority = window_authority_from_current_user_message(
        "XAUUSD را از 2026-01-01 تا 2026-03-31 با داده MT5 بررسی کن.",
        evidence_ref="request:current",
    )

    assert authority == {
        "source": "current_user_explicit",
        "evidence_ref": "request:current",
        "user_explicit": True,
        "server_owned": True,
    }


def test_server_owned_prepared_reuse_preserves_origin_and_approval() -> None:
    metadata = build_research_window_metadata(
        _config(),
        {},
        authority={
            "source": "prepared_config_reuse",
            "evidence_ref": "run:old/config.json",
            "source_run_id": "swarm-old",
            "current_run_id": "swarm-new",
            "reuse_approved": True,
            "containment_valid": True,
            "server_owned": True,
        },
    )

    authority = metadata["window_authority"]
    assert authority["source"] == "prepared_config_reuse"
    assert authority["source_run_id"] == "swarm-old"
    assert authority["current_run_id"] == "swarm-new"
    assert authority["reuse_approved"] is True


def test_copied_config_without_server_owned_authority_is_unknown_and_downgraded() -> None:
    metadata = build_research_window_metadata(
        _config(),
        {"trade_count": 31},
        authority={
            "source": "prepared_config_reuse",
            "source_run_id": "swarm-old",
            "server_owned": False,
        },
    )

    assert metadata["window_authority"]["source"] == "unknown"
    assert metadata["window_authority"]["user_explicit"] is False
    assert metadata["research_sufficiency"]["status"] == "unknown_not_enforced"
    assert metadata["official_conclusion"] == "EDGE NOT SHOWN ON LIMITED BASELINE"


def test_absent_coverage_probe_is_explicit_and_cannot_support_depth_claim() -> None:
    metadata = build_research_window_metadata(_config(), {})

    assert metadata["coverage_probe"] == {"status": "not_performed"}
    assert metadata["history_depth_claim_supported"] is False


def test_official_snapshot_row_count_populates_bars_count() -> None:
    metadata = build_research_window_metadata(
        _config(),
        {},
        snapshot_metadata={"row_count": 3518},
    )

    assert metadata["sample"]["bars_count"] == 3518


def test_worker_writable_config_cannot_assert_server_owned_window_authority(tmp_path) -> None:
    config = _config()
    config["_server_window_authority"] = {
        "source": "prepared_config_reuse",
        "source_run_id": "forged-old-run",
        "server_owned": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    card = write_run_card(tmp_path, config, {})

    assert card["research_window"]["window_authority"]["source"] == "unknown"


def test_server_owned_authority_sidecar_round_trips_without_using_config(tmp_path) -> None:
    authority = {
        "source": "prepared_config_reuse",
        "evidence_ref": "run:old/config.json",
        "source_run_id": "swarm-old",
        "current_run_id": "swarm-new",
        "reuse_approved": True,
        "containment_valid": True,
        "server_owned": True,
    }

    persist_server_window_authority(tmp_path, authority)

    assert load_server_window_authority(tmp_path) == authority


def test_post_execution_finalizer_uses_trusted_authority_not_config_claim(tmp_path) -> None:
    config = _config()
    config["_server_window_authority"] = {
        "source": "worker_generated",
        "server_owned": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_run_card(tmp_path, config, {"trade_count": 31})

    finalized = finalize_research_window_metadata(
        tmp_path,
        {
            "source": "current_user_explicit",
            "evidence_ref": "request:current#dates",
            "user_explicit": True,
            "server_owned": True,
        },
    )

    assert finalized is not None
    card = json.loads((tmp_path / "run_card.json").read_text(encoding="utf-8"))
    assert card["research_window"]["window_authority"]["source"] == "current_user_explicit"


def test_post_execution_finalizer_uses_run_card_snapshot_for_bars_count(tmp_path) -> None:
    config = _config()
    config.pop("_mt5_snapshot_provenance")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_run_card(tmp_path, config, {"trade_count": 17})
    card_path = tmp_path / "run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["mt5_snapshot"] = {"row_count": 3518}
    card_path.write_text(json.dumps(card), encoding="utf-8")

    finalized = finalize_research_window_metadata(
        tmp_path,
        {"source": "unknown", "evidence_ref": "request:current", "server_owned": True},
    )

    assert finalized is not None
    assert finalized["sample"]["bars_count"] == 3518


def test_backtest_parent_finalizes_run_card_from_internal_authority(tmp_path, monkeypatch) -> None:
    config = _config() | {"source": "yfinance"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "signal_engine.py").write_text("# test engine\n", encoding="utf-8")

    def fake_execute(_self, _entry, run_dir, **_kwargs):
        write_run_card(run_dir, config, {"trade_count": 31})
        return SimpleNamespace(success=True, exit_code=0, stdout="", stderr="", artifacts={})

    monkeypatch.setattr(backtest_tool.Runner, "execute", fake_execute)
    monkeypatch.setattr(backtest_tool, "safe_run_dir", lambda value: tmp_path)
    result = json.loads(
        backtest_tool.run_backtest(
            str(tmp_path),
            window_authority={
                "source": "current_user_explicit",
                "evidence_ref": "request:current#dates",
                "user_explicit": True,
                "server_owned": True,
            },
        )
    )

    assert result["status"] == "ok"
    card = json.loads((tmp_path / "run_card.json").read_text(encoding="utf-8"))
    assert card["research_window"]["window_authority"]["source"] == "current_user_explicit"


def _binding_with_unknown_window() -> dict:
    return {
        "symbol": "XAUUSD_o",
        "source": "mt5",
        "timeframe": "5m",
        "identity_hash": "identity",
        "config_sha256": "config",
        "strategy_sha256": "strategy",
        "strategy_logic": "Executed logic.",
        "research_window": build_research_window_metadata(_config(), {"trade_count": 31}),
    }


def test_strict_report_renders_server_generated_limited_baseline_verdict() -> None:
    report = render_bound_strict_report(_binding_with_unknown_window(), "Worker analysis only.")

    assert "## Research Sufficiency (Server-Generated)" in report
    assert "unknown_not_enforced" in report
    assert "EDGE NOT SHOWN ON LIMITED BASELINE" in report
    assert "Broad verdict authorized: `False`" in report


def test_strict_report_rejects_broad_no_edge_verdict_on_unknown_sufficiency() -> None:
    try:
        render_bound_strict_report(
            _binding_with_unknown_window(),
            "**Verdict: NO — no tradeable edge.**",
        )
    except NarrativeMismatch as exc:
        assert "unauthorized broad verdict" in str(exc)
    else:
        raise AssertionError("unauthorized broad no-edge verdict was accepted")


def test_strict_report_allows_limited_negative_baseline_language() -> None:
    report = render_bound_strict_report(
        _binding_with_unknown_window(),
        "The observed limited baseline was negative; more evidence is required.",
    )

    assert "observed limited baseline was negative" in report


def test_future_server_authorized_broad_verdict_remains_possible() -> None:
    binding = _binding_with_unknown_window()
    binding["research_window"] = {
        **binding["research_window"],
        "research_sufficiency": {"status": "sufficient"},
        "verdict_authority": "server_policy",
        "broad_verdict_authorized": True,
    }

    report = render_bound_strict_report(binding, "Verdict: NO — no tradeable edge.")

    assert "no tradeable edge" in report


def test_strict_report_blocks_unsupported_history_depth_claim_without_probe() -> None:
    try:
        render_bound_strict_report(
            _binding_with_unknown_window(),
            "M5 depth defines the window.",
        )
    except NarrativeMismatch as exc:
        assert "unsupported history-depth claim" in str(exc)
    else:
        raise AssertionError("unsupported history-depth claim was accepted")


def test_strict_report_blocks_terminal_depth_binding_claim_without_probe() -> None:
    try:
        render_bound_strict_report(
            _binding_with_unknown_window(),
            "The 5M terminal depth binding defines this window.",
        )
    except NarrativeMismatch as exc:
        assert "unsupported history-depth claim" in str(exc)
    else:
        raise AssertionError("terminal depth binding claim was accepted")


def test_strict_report_blocks_persian_history_depth_claim_without_probe() -> None:
    try:
        render_bound_strict_report(
            _binding_with_unknown_window(),
            "عمق تاریخچه M5 پنجره را محدود کرده است.",
        )
    except NarrativeMismatch as exc:
        assert "unsupported history-depth claim" in str(exc)
    else:
        raise AssertionError("unsupported Persian history-depth claim was accepted")
