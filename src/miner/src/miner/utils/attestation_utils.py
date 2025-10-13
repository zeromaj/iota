from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

from loguru import logger

from common.models.api_models import AttestationChallengeResponse, MinerAttestationPayload


class AttestationUnavailableError(RuntimeError):
    """Raised when the native attestation helper cannot be used."""

    def __init__(self, message: str, *, error_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code


_ATTEST_MODULE: Optional[ModuleType] = None


def _import_attest_module() -> Optional[ModuleType]:
    global _ATTEST_MODULE
    if _ATTEST_MODULE is not None:
        return _ATTEST_MODULE

    module_name = "common.attest"
    try:
        _ATTEST_MODULE = importlib.import_module(module_name)
        return _ATTEST_MODULE
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError):
            logger.warning("Native attestation module %s not found on PYTHONPATH", module_name)
            return None
        logger.exception("Unexpected error importing %s", module_name)
        return None


def collect_attestation_payload(challenge: AttestationChallengeResponse) -> MinerAttestationPayload:
    """Run the native attestation helper and map its output to the API payload."""
    attest_module = _import_attest_module()
    if attest_module is None:
        raise AttestationUnavailableError("native attestation helper unavailable")

    try:
        attestation_output = attest_module.collect(challenge.challenge_blob)
    except Exception as exc:  # pragma: no cover - native module bubbles detailed errors
        error_code: int | None = None
        if exc.args:
            first_arg = exc.args[0]
            if isinstance(first_arg, int):
                error_code = first_arg
        raise AttestationUnavailableError(
            "native attestation helper raised an exception",
            error_code=error_code,
        ) from exc

    if not isinstance(attestation_output, dict):
        raise AttestationUnavailableError("attestation helper returned unexpected output")

    try:
        payload_blob = attestation_output["payload_blob"]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttestationUnavailableError("attestation helper output missing field") from exc

    if isinstance(payload_blob, (bytes, bytearray)):
        try:
            payload_blob = payload_blob.decode("ascii")
        except Exception as exc:  # pragma: no cover - defensive guard
            raise AttestationUnavailableError("attestation helper returned malformed blob") from exc
    if not isinstance(payload_blob, str):
        raise AttestationUnavailableError("attestation helper returned malformed blob")

    payload = MinerAttestationPayload(payload_blob=payload_blob)
    logger.debug("Collected attestation blob")
    return payload
