"""Run-scoped creation and evidence-backed enrichment of ExecutionIdentity."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.execution_identity import (
    AttachmentAssertionType,
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    FallbackPolicy,
    IdentityProvenance,
    ProvenanceAuthority,
    SourceMode,
    SyntheticDataPolicy,
)


EXECUTION_IDENTITY_ARTIFACT = "execution_identity.json"
_EXPLICIT_SOURCE_RE = re.compile(r"\bsource\s*(?:=|:)\s*['\"]?([a-z0-9_-]+)", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,11}(?:[./_-][A-Z0-9]{1,11})?\b")
_DECLARED_FIELD_RE = re.compile(
    r"(?:^|\n)\s*(?:required\s+)?(symbol|instrument|asset|source|platform)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE,
)
_REQUIREMENT_WORD_RE = re.compile(r"\b(?:must|required|only|use)\b", re.IGNORECASE)


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

    def persist(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / EXECUTION_IDENTITY_ARTIFACT
        path.write_text(self._identity.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _from_user_message(user_message: str) -> ExecutionIdentity:
        source_match = _EXPLICIT_SOURCE_RE.search(user_message or "")
        source = source_match.group(1).casefold() if source_match else None
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
    for match in _SYMBOL_RE.finditer(value or ""):
        candidate = match.group(0).upper()
        if candidate not in {"PDF", "MT5", "MQL5", "HTTP", "UTC"}:
            return candidate
    return None


def _source_value(value: str) -> str | None:
    candidate = value.strip().casefold()
    return candidate if candidate else None


def _platform_value(value: str) -> str | None:
    candidate = value.strip().casefold()
    return candidate if candidate else None
