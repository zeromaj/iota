"""Shared authentication utilities for API endpoints."""

import settings
from typing import Annotated
from fastapi import HTTPException, Header, Depends, Request
from utils.epistula import EpistulaHeaders
from utils.bt_utils import verify_entity_type
from orchestrator.orchestrator import orchestrator
from slowapi.util import get_remote_address


class AuthenticatedRequest:
    """Container for validated authentication data."""

    def __init__(self, signed_by: str, headers: EpistulaHeaders):
        self.signed_by = signed_by
        self.headers = headers


async def validate_authenticated_request(
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    orchestrator_time: Annotated[str, Header(alias="X-Orchestrator-Version")],
    spec_version: Annotated[str, Header(alias="X-Spec-Version")],
) -> AuthenticatedRequest:
    """Dependency that validates Epistula headers and orchestrator version."""
    # Validate orchestrator version
    if orchestrator_time != orchestrator.orchestrator_time:
        raise HTTPException(
            status_code=409,
            detail=f"Orchestrator version mismatch. Expected: {orchestrator.orchestrator_time}, Received: {orchestrator_time}",
        )

    # Validate spec version
    if spec_version != str(settings.__spec_version__):
        raise HTTPException(
            status_code=409,
            detail=f"Spec version mismatch. Expected: {settings.__spec_version__}, Received: {spec_version}",
        )

    # Create and return headers object
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )

    return AuthenticatedRequest(signed_by=signed_by, headers=headers)


async def validate_miner_request(
    auth: AuthenticatedRequest = Depends(validate_authenticated_request),
) -> AuthenticatedRequest:
    """Dependency that validates the request is from a miner."""
    if settings.BITTENSOR:
        verify_entity_type(signed_by=auth.signed_by, metagraph=orchestrator.metagraph, required_type="miner")
    return auth


async def validate_validator_request(
    auth: AuthenticatedRequest = Depends(validate_authenticated_request),
) -> AuthenticatedRequest:
    """Dependency that validates the request is from a validator."""
    if settings.BITTENSOR:
        verify_entity_type(signed_by=auth.signed_by, metagraph=orchestrator.metagraph, required_type="validator")
    return auth


def validate_orchestrator_time(received_version: str, current_version: str):
    """Validate that the received orchestrator version matches the current version."""
    if received_version != current_version:
        raise HTTPException(
            status_code=409,
            detail=f"Orchestrator version mismatch. Expected: {current_version}, Received: {received_version}",
        )


def get_signed_by_key(request: Request) -> str:
    """Get the signed_by key for rate limiting. Falls back to IP address if not authenticated."""
    try:
        # Try to get signed_by from request headers
        signed_by = request.headers.get("Epistula-Signed-By")
        if signed_by:
            return signed_by
    except Exception:
        pass
    # Fall back to IP address for unauthenticated endpoints
    return get_remote_address(request)
