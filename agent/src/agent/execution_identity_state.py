"""Run-scoped creation and evidence-backed enrichment of ExecutionIdentity."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.agent.current_user_intent import explicit_source_from_current_user_message
from src.execution_identity import (
    AttachmentAssertionType,
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionResolution,
    FallbackPolicy,
    IdentityConflict,
    IdentityProvenance,
    ProvenanceAuthority,
    SourceMode,
    SyntheticDataPolicy,
)


EXECUTION_IDENTITY_ARTIFACT = "execution_identity.json"
_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,11}(?:[./_-][A-Z0-9]{1,11})?\b")
_DECLARED_SYMBOL_RE = re.compile(
    r"(?:^|\n)\s*(?:requested\s+)?(?:symbol|ticker|instrument)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE,
)
_DECLARED_FIELD_RE = re.compile(
    r"(?:^|\n)\s*(?:required\s+)?(symbol|instrument|asset|source|platform)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE,
)
_REQUIREMENT_WORD_RE = re.compile(r"\b(?:must|required|only|use)\b", re.IGNORECASE)
_QUOTE_CURRENCY_SUFFIXES = ("USDT", "USDC", "USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD")


class ExecutionIdentityLedger:
    """Own the current run's identity revisions, never model-generated prose."""

    def __init__(self, *, run_dir: Path, user_message: str) -> None:
        self.run_dir = Path(run_dir)
        self._identity = self._from_user_message(user_message)
        self.persist()

    @property
    def identity(self) -> ExecutionIdentity:
        return self._identity

    def snapshot(self) -> ExecutionIdentity:
        """Return the immutable current revision for an execution boundary."""
        return self._identity

    def ingest_document_result(self, result: str, *, call_id: str) -> None:
        """Add only explicit document declarations as attachment requirements.

        Unlabelled mentions deliberately remain references: attachment text may
        describe examples, prior research, or alternatives and cannot become an
        execution constraint just because it contains a ticker-like token.
        """
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, Mapping) or payload.get("status") not in {"ok", "success"}:
            return
        text = str(payload.get("text") or "")
        file_ref = str(payload.get("file") or "attachment")
        fields = {match.group(1).casefold(): match.group(2).strip() for match in _DECLARED_FIELD_RE.finditer(text)}
        if not fields:
            return

        symbol = _first_symbol(fields.get("symbol") or fields.get("instrument") or fields.get("asset") or "")
        source = _source_value(fields.get("source") or "")
        platform = _platform_value(fields.get("platform") or "")
        requirement = bool(symbol or source or platform)
        if not requirement:
            return
        provenance = IdentityProvenance(
            authority=ProvenanceAuthority.CURRENT_ATTACHMENT,
            origin="read_document",
            evidence_ref=f"{file_ref}#tool={call_id}",
            assertion_type=AttachmentAssertionType.REQUIREMENT,
            confidence=1.0,
        )
        self._merge_current_requirement(
            symbol=symbol,
            source=source,
            platform=platform,
            provenance=provenance,
        )
        self.persist()

    def ingest_resolver_result(
        self,
        *,
        arguments: Mapping[str, Any],
        result: str,
        call_id: str,
        success: bool,
    ) -> None:
        """Append only a source-native resolution to a strict identity."""
        if self._identity.mode is not ExecutionMode.SOURCE_SCOPED:
            return
        if not success:
            self._reject_unresolved_identity()
            return
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            self._reject_unresolved_identity()
            return
        data = payload.get("data") if isinstance(payload, Mapping) else None
        candidates = data.get("candidates") if isinstance(data, Mapping) else None
        if not isinstance(candidates, list) or len(candidates) != 1 or not self._identity.requests:
            self._reject_unresolved_identity()
            return
        candidate = candidates[0] if isinstance(candidates[0], Mapping) else {}
        request = self._identity.requests[0]
        existing = next((item for item in self._identity.resolutions if item.request_id == request.request_id), None)
        source = str(candidate.get("source") or arguments.get("source") or "").casefold() or None
        requested = str(candidate.get("requested_symbol") or candidate.get("symbol") or "").upper() or None
        if request.source and source != request.source.casefold():
            self._reject_or_record_conflict(
                existing=existing,
                field="source",
                existing_value=request.source,
                incoming_value=source,
                call_id=call_id,
                message="resolver source conflicts with the source-scoped request",
            )
            return
        if request.symbol and requested != request.symbol.upper():
            self._reject_or_record_conflict(
                existing=existing,
                field="requested_symbol",
                existing_value=request.symbol,
                incoming_value=requested,
                call_id=call_id,
                message="resolver requested symbol conflicts with the source-scoped request",
            )
            return
        incoming = ExecutionResolution(
            request_id=request.request_id,
            canonical_asset=str(candidate.get("name") or "") or None,
            # Preserve broker-native spelling in evidence; comparison-only
            # boundaries normalize separately.
            resolved_symbol=str(candidate.get("resolved_symbol") or "").strip() or None,
            source=source,
            broker=str(candidate.get("source_namespace") or "") or None,
            venue=str(candidate.get("exchange") or "") or None,
            asset_class=str(candidate.get("type") or "") or None,
            market=str(candidate.get("market_type") or candidate.get("market") or "") or None,
            resolver_evidence_ref=f"tool:search_symbol:{call_id}",
            provenance=(
                IdentityProvenance(
                    authority=ProvenanceAuthority.SOURCE_RESOLVER,
                    origin="search_symbol",
                    evidence_ref=f"tool:search_symbol:{call_id}",
                    confidence=1.0,
                ),
            ),
        )
        if existing is not None:
            mismatch = next(
                (
                    (field, getattr(existing, field), getattr(incoming, field))
                    for field in (
                        "canonical_asset",
                        "resolved_symbol",
                        "source",
                        "broker",
                        "venue",
                        "asset_class",
                        "market",
                    )
                    if getattr(existing, field) != getattr(incoming, field)
                ),
                None,
            )
            if mismatch is None:
                # A repeatable search_symbol confirmation has a different tool
                # call ID but is not a new identity revision.
                return
            field, existing_value, incoming_value = mismatch
            self._record_conflict(
                field=field,
                existing_value=existing_value,
                incoming_value=incoming_value,
                call_id=call_id,
                message=f"resolver result conflicts with verified {field}",
            )
            return
        self._identity = self._identity.with_resolution(incoming)
        self.persist()

    def _reject_unresolved_identity(self) -> None:
        """Reject an unusable resolver result only before verification."""
        if self._identity.resolutions:
            return
        self._identity = self._identity.with_status(ExecutionIdentityStatus.REJECTED)
        self.persist()

    def _reject_or_record_conflict(
        self,
        *,
        existing: ExecutionResolution | None,
        field: str,
        existing_value: str | None,
        incoming_value: str | None,
        call_id: str,
        message: str,
    ) -> None:
        if existing is None:
            self._identity = self._identity.with_status(ExecutionIdentityStatus.REJECTED)
            self.persist()
            return
        self._record_conflict(
            field=field,
            existing_value=existing_value,
            incoming_value=incoming_value,
            call_id=call_id,
            message=message,
        )

    def _record_conflict(
        self,
        *,
        field: str,
        existing_value: str | None,
        incoming_value: str | None,
        call_id: str,
        message: str,
    ) -> None:
        self._identity = self._identity.with_conflict(
            IdentityConflict(
                field=field,
                existing_value=existing_value,
                incoming_value=incoming_value,
                provenance=IdentityProvenance(
                    authority=ProvenanceAuthority.SOURCE_RESOLVER,
                    origin="search_symbol",
                    evidence_ref=f"tool:search_symbol:{call_id}",
                    confidence=1.0,
                ),
                message=message,
            )
        )
        self.persist()

    def persist(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / EXECUTION_IDENTITY_ARTIFACT
        path.write_text(self._identity.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _from_user_message(user_message: str) -> ExecutionIdentity:
        source = explicit_source_from_current_user_message(user_message)
        symbol = _first_symbol(user_message)
        if source:
            policy = ExecutionPolicy(
                source_mode=SourceMode.STRICT,
                fallback=FallbackPolicy.DENY,
                cross_source_fallback=False,
                synthetic=SyntheticDataPolicy.FORBID,
            )
            mode = ExecutionMode.SOURCE_SCOPED
        else:
            policy = ExecutionPolicy()
            mode = ExecutionMode.GENERIC
        requests = ()
        if symbol or source:
            requests = (
                ExecutionRequest(
                    request_id="request-1",
                    symbol=symbol,
                    source=source,
                    provenance=(
                        IdentityProvenance(
                            authority=ProvenanceAuthority.CURRENT_USER,
                            origin="user_message",
                            confidence=1.0,
                        ),
                    ),
                ),
            )
        return ExecutionIdentity(
            identity_id=f"execution-{uuid4().hex}",
            mode=mode,
            status=(
                ExecutionIdentityStatus.PARTIALLY_SPECIFIED
                if requests
                else ExecutionIdentityStatus.UNRESOLVED
            ),
            requests=requests,
            policy=policy,
        )

    def _merge_current_requirement(
        self,
        *,
        symbol: str | None,
        source: str | None,
        platform: str | None,
        provenance: IdentityProvenance,
    ) -> None:
        if self._identity.requests:
            existing = self._identity.requests[0]
            if any(
                old and new and old.casefold() != new.casefold()
                for old, new in ((existing.symbol, symbol), (existing.source, source), (existing.platform, platform))
            ):
                self._identity = self._identity.model_copy(
                    update={"status": ExecutionIdentityStatus.CONFLICT, "revision": self._identity.revision + 1}
                )
                return
            request = existing.model_copy(
                update={
                    "symbol": existing.symbol or symbol,
                    "source": existing.source or source,
                    "platform": existing.platform or platform,
                    "provenance": (*existing.provenance, provenance),
                }
            )
        else:
            request = ExecutionRequest(
                request_id="request-1",
                symbol=symbol,
                source=source,
                platform=platform,
                provenance=(provenance,),
            )
        strict = bool(request.source or request.platform)
        self._identity = self._identity.model_copy(
            update={
                "revision": self._identity.revision + 1,
                "mode": ExecutionMode.SOURCE_SCOPED if strict else ExecutionMode.GENERIC,
                "status": ExecutionIdentityStatus.PARTIALLY_SPECIFIED,
                "requests": (request,),
                "policy": (
                    ExecutionPolicy(
                        source_mode=SourceMode.STRICT,
                        fallback=FallbackPolicy.DENY,
                        cross_source_fallback=False,
                        synthetic=SyntheticDataPolicy.FORBID,
                    )
                    if strict
                    else self._identity.policy
                ),
            }
        )


def _first_symbol(value: str) -> str | None:
    """Return an explicit or structurally credible requested symbol.

    Execution identity must fail unresolved rather than mistake document titles,
    indicator names, or all-caps prose (for example ``VWAP``) for an instrument.
    A labelled current-task declaration wins; the fallback accepts only a
    delimiter-qualified identifier or a conventional quote-currency pair.
    """
    declared = _DECLARED_SYMBOL_RE.search(value or "")
    candidates = _SYMBOL_RE.finditer(declared.group(1) if declared else value or "")
    for match in candidates:
        candidate = match.group(0).upper()
        if candidate not in {"PDF", "MT5", "MQL5", "HTTP", "UTC"}:
            if declared:
                return candidate
            if any(separator in candidate for separator in (".", "/", "_", "-")):
                return candidate
            if candidate.endswith(_QUOTE_CURRENCY_SUFFIXES):
                return candidate
    return None


def _source_value(value: str) -> str | None:
    candidate = value.strip().casefold()
    return candidate if candidate else None


def _platform_value(value: str) -> str | None:
    candidate = value.strip().casefold()
    return candidate if candidate else None
