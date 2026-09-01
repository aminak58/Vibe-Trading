"""Unit coverage for server-owned execution identity contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.execution_identity import (
    AttachmentAssertionType,
    ExecutionIdentity,
    ExecutionIdentityStatus,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionRequest,
    ExecutionResolution,
    FallbackPolicy,
    IdentityProvenance,
    ProvenanceAuthority,
    SourceMode,
)


def _mt5_identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        identity_id="identity-1",
        mode=ExecutionMode.SOURCE_SCOPED,
        status=ExecutionIdentityStatus.RESOLVING,
        policy=ExecutionPolicy(
            source_mode=SourceMode.STRICT,
            fallback=FallbackPolicy.DENY,
            cross_source_fallback=False,
        ),
        requests=(
            ExecutionRequest(
                request_id="gold",
                symbol="XAUUSD",
                source="mt5",
                provenance=(
                    IdentityProvenance(
                        authority=ProvenanceAuthority.CURRENT_USER,
                        origin="user_message",
                    ),
                ),
            ),
        ),
    )


def test_identity_hash_is_deterministic_and_excludes_volatile_evidence() -> None:
    first = _mt5_identity()
    second = first.model_copy(
        update={
            "requests": (
                first.requests[0].model_copy(
                    update={
                        "provenance": (
                            IdentityProvenance(
                                authority=ProvenanceAuthority.CURRENT_USER,
                                origin="user_message",
                                evidence_ref="different-page-location",
                            ),
                        )
                    }
                ),
            )
        }
    )

    assert first.identity_hash == second.identity_hash


def test_requested_fields_are_immutable() -> None:
    identity = _mt5_identity()

    with pytest.raises(ValidationError):
        identity.requests[0].symbol = "XAUT-USDT"  # type: ignore[misc]


def test_resolution_is_append_only_and_scoped_to_one_request() -> None:
    identity = _mt5_identity()
    resolution = ExecutionResolution(
        request_id="gold",
        resolved_symbol="XAUUSD_o",
        source="mt5",
        resolver_evidence_ref="tool:search_symbol:1",
    )

    verified = identity.with_resolution(resolution)

    assert verified.status is ExecutionIdentityStatus.VERIFIED
    assert verified.revision == 2
    assert verified.resolutions[0].resolved_symbol == "XAUUSD_o"
    with pytest.raises(ValueError, match="cannot be silently overwritten"):
        verified.with_resolution(resolution)


def test_strict_source_rejects_cross_source_resolution() -> None:
    identity = _mt5_identity()

    with pytest.raises(ValidationError, match="requested source namespace"):
        identity.with_resolution(
            ExecutionResolution(
                request_id="gold",
                resolved_symbol="XAUT-USDT",
                source="okx",
                resolver_evidence_ref="tool:search_symbol:2",
            )
        )


def test_strict_policy_is_fail_closed() -> None:
    with pytest.raises(ValidationError, match="strict source mode"):
        ExecutionPolicy(source_mode=SourceMode.STRICT)


def test_attachment_assertion_type_is_preserved_in_provenance() -> None:
    provenance = IdentityProvenance(
        authority=ProvenanceAuthority.CURRENT_ATTACHMENT,
        origin="read_document",
        assertion_type=AttachmentAssertionType.REQUIREMENT,
        evidence_ref="upload.pdf#page=2",
    )

    assert provenance.assertion_type is AttachmentAssertionType.REQUIREMENT
