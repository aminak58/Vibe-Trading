"""Immutable, parent-acquired MT5 bar snapshots for sandboxed backtests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SNAPSHOT_SCHEMA_VERSION = "mt5-bar-snapshot/v3"
MANIFEST_RELATIVE_PATH = Path("data/mt5_snapshot_manifest.json")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
TIMESTAMP_TIMEZONE = "UTC"
TIMESTAMP_ENCODING = "unix_epoch_seconds"
TIMESTAMP_REPRESENTATION = "pandas-datetimeindex-utc"
TIMEZONE_CONFIDENCE = "official_mt5_sdk_utc"


class MT5SnapshotError(ValueError):
    """Raised when a broker-backed MT5 snapshot cannot be trusted."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "mt5_snapshot_invalid_schema",
        stage: str = "snapshot_validation",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.stage = stage
        self.details = dict(details or {})
        super().__init__(f"MT5-backed data acquisition/handoff failure: [{error_code}] {message}")


def _fail(
    message: str,
    *,
    error_code: str = "mt5_snapshot_invalid_schema",
    stage: str = "snapshot_validation",
    details: dict[str, Any] | None = None,
) -> MT5SnapshotError:
    return MT5SnapshotError(message, error_code=error_code, stage=stage, details=details)


def _diagnostic_details(
    config: dict[str, Any],
    *,
    timeframe: str | None = None,
    resolved_symbol: str | None = None,
) -> dict[str, Any]:
    codes = config.get("codes") or []
    requested_symbol = codes[0].strip() if len(codes) == 1 and isinstance(codes[0], str) else None
    return {
        "requested_symbol": requested_symbol,
        "codes": list(codes) if isinstance(codes, list) else [],
        "resolved_symbol": resolved_symbol,
        "interval": str(config.get("interval") or "1D"),
        "timeframe": timeframe,
        "requested_start": str(config.get("start_date") or ""),
        "requested_end": str(config.get("end_date") or ""),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_relpath(symbol: str, interval: str) -> Path:
    safe_symbol = "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)
    safe_interval = "".join(c if c.isalnum() or c in "-_" else "_" for c in interval)
    return Path("data") / f"{safe_symbol}_{safe_interval}_mt5_snapshot.csv"


def _spread_evidence_relpath(symbol: str, interval: str) -> Path:
    safe_symbol = "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)
    safe_interval = "".join(c if c.isalnum() or c in "-_" else "_" for c in interval)
    return Path("data") / f"{safe_symbol}_{safe_interval}_mt5_spread_evidence.csv"


def _is_utc_index(index: pd.DatetimeIndex) -> bool:
    return index.tz is not None and str(index.tz) == TIMESTAMP_TIMEZONE


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise _fail("MT5 returned no usable bars")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise _fail(f"snapshot is missing required columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise _fail("snapshot timestamps are not a DatetimeIndex")
    if not _is_utc_index(frame.index):
        raise _fail("snapshot timestamps are not timezone-aware UTC")
    if frame.index.hasnans or not frame.index.is_monotonic_increasing:
        raise _fail("snapshot timestamps are invalid or not monotonic")
    if frame.index.has_duplicates:
        raise _fail("snapshot contains duplicate timestamps")
    if frame.loc[:, list(REQUIRED_COLUMNS)].isna().any().any():
        raise _fail("snapshot contains null OHLCV values")
    return frame.loc[:, list(REQUIRED_COLUMNS)].copy()


def _load_utc_csv_snapshot(snapshot_path: Path) -> pd.DataFrame:
    """Load the strict v2 CSV format without reinterpreting naïve timestamps."""
    try:
        raw = pd.read_csv(snapshot_path)
    except (OSError, ValueError) as exc:
        raise _fail("snapshot file cannot be loaded") from exc
    if "timestamp" not in raw.columns:
        raise _fail("snapshot file is missing timestamp column")

    timestamp_strings = raw.pop("timestamp").astype(str)
    # CSV must encode UTC explicitly.  Parsing a naïve legacy string with
    # utc=True would silently reinterpret it, which strict snapshot mode forbids.
    if not timestamp_strings.str.endswith(("+00:00", "Z")).all():
        raise _fail("snapshot timestamps are not explicitly UTC encoded")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(timestamp_strings, utc=True), name="timestamp")
    except (TypeError, ValueError) as exc:
        raise _fail("snapshot timestamps cannot be parsed as UTC") from exc
    raw.index = index
    return raw


def _broker_server_label(mt5: Any) -> str | None:
    try:
        account = mt5.account_info()
        server = getattr(account, "server", None)
        return str(server) if server else None
    except Exception:
        return None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_mt5_snapshot(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Acquire one explicit-MT5 snapshot in the trusted parent process.

    A pre-existing manifest is reused only after full integrity validation. This
    gives an identical rerun the same immutable input rather than a fresh live
    terminal fetch.
    """
    if config.get("source") != "mt5":
        raise _fail("snapshot preparation requires source='mt5'")
    manifest_path = run_dir / MANIFEST_RELATIVE_PATH
    if manifest_path.exists():
        _frame, manifest = load_mt5_snapshot(run_dir, config)
        return manifest
    if (run_dir / "data").exists() and any((run_dir / "data").iterdir()):
        raise _fail("data directory exists without a valid MT5 snapshot manifest")

    codes = config.get("codes") or []
    if len(codes) != 1 or not isinstance(codes[0], str) or not codes[0].strip():
        raise _fail("explicit MT5 snapshot mode currently requires one non-empty symbol")
    requested_symbol = codes[0].strip()
    interval = str(config.get("interval") or "1D")
    start_date = str(config.get("start_date") or "")
    end_date = str(config.get("end_date") or "")
    if not start_date or not end_date:
        raise _fail("requested start_date and end_date are required")

    from backtest.loaders import mt5_loader

    mt5 = mt5_loader._import_mt5()
    if mt5 is None or not mt5_loader._ensure_initialized():
        raise _fail(
            "MT5 terminal is unavailable in the trusted parent process.",
            error_code="mt5_terminal_unavailable",
            stage="terminal_attach",
            details=_diagnostic_details(config),
        )
    timeframe_name = mt5_loader._INTERVAL_MAP.get(interval)
    if timeframe_name is None:
        raise _fail(
            f"unsupported MT5 timeframe {interval!r}",
            error_code="mt5_snapshot_invalid_schema",
            stage="timeframe_validation",
            details=_diagnostic_details(config),
        )
    resolved_symbol = mt5_loader._resolve_broker_symbol(mt5, requested_symbol)
    if not resolved_symbol:
        raise _fail(
            f"broker symbol resolution failed for {requested_symbol!r}",
            error_code="mt5_symbol_resolution_failed",
            stage="symbol_resolution",
            details=_diagnostic_details(config, timeframe=timeframe_name),
        )

    loader = mt5_loader.DataLoader()
    try:
        raw_frame = loader._fetch_one(
            requested_symbol,
            start_date,
            end_date,
            timeframe_name,
            include_spread=True,
            diagnostic=True,
            interval=interval,
        )
    except mt5_loader.MT5AcquisitionError as exc:
        details = _diagnostic_details(config, timeframe=timeframe_name, resolved_symbol=resolved_symbol)
        details.update(exc.details)
        raise _fail(
            str(exc),
            error_code=exc.error_code,
            stage=exc.stage,
            details=details,
        ) from exc
    try:
        frame = _validate_frame(raw_frame)
    except MT5SnapshotError as exc:
        details = _diagnostic_details(
            config,
            timeframe=timeframe_name,
            resolved_symbol=resolved_symbol,
        )
        details.update(exc.details)
        raise _fail(
            exc.message,
            error_code=exc.error_code,
            stage=exc.stage,
            details=details,
        ) from exc
    symbol_info = mt5.symbol_info(resolved_symbol)
    if symbol_info is None:
        raise _fail("broker symbol metadata is unavailable for cost normalization")
    from backtest.engines.forex import _normalize_symbol, _pip_value
    from backtest.mt5_cost_model import MT5CostModelError, resolve_cost_model

    try:
        cost_model, spread_evidence = resolve_cost_model(
            config.get("cost_model"),
            frame=raw_frame,
            point=float(getattr(symbol_info, "point", 0.0) or 0.0),
            engine_pip_size=_pip_value(_normalize_symbol(requested_symbol)),
            requested_symbol=requested_symbol,
            resolved_symbol=resolved_symbol,
        )
    except MT5CostModelError as exc:
        raise _fail(str(exc)) from exc
    # The caller persists this resolved form before sandbox launch.  Reuse of
    # an existing manifest therefore checks the exact same deterministic model.
    config["cost_model"] = cost_model

    relative_snapshot = _snapshot_relpath(requested_symbol, interval)
    snapshot_path = run_dir / relative_snapshot
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        raise _fail("snapshot file already exists without a valid manifest")
    evidence_path: Path | None = None
    try:
        frame.to_csv(snapshot_path, index=True, index_label="timestamp", lineterminator="\n")
        content_hash = _sha256(snapshot_path)
        if spread_evidence is not None:
            relative_evidence = _spread_evidence_relpath(requested_symbol, interval)
            evidence_path = run_dir / relative_evidence
            spread_evidence.to_csv(evidence_path, index=True, index_label="timestamp", lineterminator="\n")
            cost_model["spread"]["evidence"].update({
                "path": relative_evidence.as_posix(),
                "sha256": _sha256(evidence_path),
            })
    except OSError as exc:
        raise _fail(f"could not persist immutable snapshot: {exc}") from exc

    manifest: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": "mt5",
        "requested_symbol": requested_symbol,
        "resolved_symbol": resolved_symbol,
        "broker_server": _broker_server_label(mt5),
        "timeframe": interval,
        "requested_start": start_date,
        "requested_end": end_date,
        "actual_start": frame.index[0].isoformat(),
        "actual_end": frame.index[-1].isoformat(),
        "row_count": int(len(frame)),
        "columns": list(REQUIRED_COLUMNS),
        "timestamp_timezone": TIMESTAMP_TIMEZONE,
        "timestamp_encoding": TIMESTAMP_ENCODING,
        "timestamp_representation": TIMESTAMP_REPRESENTATION,
        "timezone_confidence": TIMEZONE_CONFIDENCE,
        "cost_model": cost_model,
        "snapshot_path": relative_snapshot.as_posix(),
        "sha256": content_hash,
        "acquired_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        _write_manifest(manifest_path, manifest)
    except OSError as exc:
        try:
            snapshot_path.unlink(missing_ok=True)
            if evidence_path is not None:
                evidence_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise _fail(f"could not persist snapshot manifest: {exc}") from exc
    return manifest


def load_mt5_snapshot(run_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify and load a parent-created MT5 snapshot without touching loaders."""
    manifest_path = run_dir / MANIFEST_RELATIVE_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _fail("snapshot manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise _fail("snapshot manifest version is unsupported")
    if manifest.get("source") != "mt5":
        raise _fail("snapshot manifest source is not mt5")
    if manifest.get("timestamp_timezone") != TIMESTAMP_TIMEZONE:
        raise _fail("snapshot manifest timestamp timezone is unsupported")
    if manifest.get("timestamp_encoding") != TIMESTAMP_ENCODING:
        raise _fail("snapshot manifest timestamp encoding is unsupported")
    if manifest.get("timestamp_representation") != TIMESTAMP_REPRESENTATION:
        raise _fail("snapshot manifest timestamp representation is unsupported")
    if manifest.get("timezone_confidence") != TIMEZONE_CONFIDENCE:
        raise _fail("snapshot manifest timezone confidence is unsupported")
    from backtest.mt5_cost_model import MT5CostModelError, validate_resolved_cost_model, validate_spread_evidence_file
    try:
        cost_model = validate_resolved_cost_model(manifest.get("cost_model"))
        validate_spread_evidence_file(run_dir, cost_model, _sha256)
    except MT5CostModelError as exc:
        raise _fail(str(exc)) from exc
    if config.get("cost_model") != cost_model:
        raise _fail("snapshot cost_model does not match backtest config")
    expected_symbol = (config.get("codes") or [None])[0]
    if manifest.get("requested_symbol") != expected_symbol:
        raise _fail("snapshot requested symbol does not match backtest config")
    for key in ("resolved_symbol", "sha256", "snapshot_path"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise _fail(f"snapshot manifest is missing {key}")
    if manifest.get("timeframe") != str(config.get("interval") or "1D"):
        raise _fail("snapshot timeframe does not match backtest config")
    if manifest.get("requested_start") != str(config.get("start_date") or "") or manifest.get("requested_end") != str(config.get("end_date") or ""):
        raise _fail("snapshot requested range does not match backtest config")
    relative = Path(manifest["snapshot_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise _fail("snapshot path is invalid")
    snapshot_path = run_dir / relative
    if not snapshot_path.is_file():
        raise _fail("snapshot file is missing")
    if _sha256(snapshot_path) != manifest["sha256"]:
        raise _fail("snapshot content hash mismatch")
    frame = _load_utc_csv_snapshot(snapshot_path)
    frame = _validate_frame(frame)
    if int(manifest.get("row_count", -1)) != len(frame):
        raise _fail("snapshot row count does not match manifest")
    if manifest.get("columns") != list(REQUIRED_COLUMNS):
        raise _fail("snapshot schema does not match manifest")
    return frame, manifest
