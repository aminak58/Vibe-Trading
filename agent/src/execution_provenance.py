"""Structured provenance checks for source-scoped quantitative results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.execution_identity import ExecutionIdentity, ExecutionMode, SourceMode


@dataclass(frozen=True)
class ProvenanceValidation:
    status: str
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "provenance_conflict"


def validate_execution_provenance(
    identity: ExecutionIdentity | None,
    artifact: Mapping[str, Any] | None,
    *,
    owning_run_id: str | None,
) -> ProvenanceValidation:
    """Compare persisted artifact provenance to strict execution identity."""
    if (
        identity is None
        or identity.mode is not ExecutionMode.SOURCE_SCOPED
        or identity.policy.source_mode is not SourceMode.STRICT
    ):
        return ProvenanceValidation("not_required")
    if not isinstance(artifact, Mapping):
        return ProvenanceValidation("provenance_conflict", ("missing authoritative acquisition artifact",))
    issues: list[str] = []
    if artifact.get("identity_hash") != identity.identity_hash:
        issues.append("identity_hash mismatch")
    if owning_run_id and artifact.get("owning_run_id") != owning_run_id:
        issues.append("owning_run_id mismatch")
    resolution = identity.resolutions[0] if identity.resolutions else None
    request = identity.requests[0] if identity.requests else None
    if request and artifact.get("requested_symbol") != request.symbol:
        issues.append("requested_symbol mismatch")
    if resolution and artifact.get("resolved_symbol") != resolution.resolved_symbol:
        issues.append("resolved_symbol mismatch")
    if request and artifact.get("effective_source") != request.source:
        issues.append("effective_source mismatch")
    if artifact.get("synthetic") is True:
        issues.append("synthetic data is forbidden for strict execution")
    if artifact.get("fallback_used") is True:
        issues.append("fallback is forbidden for strict execution")
    return ProvenanceValidation("provenance_conflict", tuple(issues)) if issues else ProvenanceValidation("passed")
