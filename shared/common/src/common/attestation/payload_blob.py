from __future__ import annotations

import base64
import struct
from dataclasses import dataclass

from common.models.api_models import MinerAttestationRuntime

_INNER_VERSION = 0x01
_WRAPPED_VERSION = 0x02
_UINT32 = struct.Struct("<I")
_FLOAT64 = struct.Struct("<d")


class MinerAttestationPayloadDecodeError(ValueError):
    """Raised when an attestation payload blob cannot be decoded."""


@dataclass(slots=True)
class MinerAttestationPayloadBundle:
    nonce: str
    challenge_id: str
    heartbeat_nonce: str
    artifact: str
    runtime: MinerAttestationRuntime | None = None


def _read_string(raw: memoryview, index: int) -> tuple[str, int]:
    if index + _UINT32.size > len(raw):
        raise MinerAttestationPayloadDecodeError("unexpected end of payload blob")
    (length,) = _UINT32.unpack_from(raw, index)
    index += _UINT32.size
    end = index + length
    if end > len(raw):
        raise MinerAttestationPayloadDecodeError("payload blob truncated")
    try:
        text = raw[index:end].tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - defensive guard
        raise MinerAttestationPayloadDecodeError("payload blob contained invalid UTF-8") from exc
    return text, end


def _write_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return _UINT32.pack(len(data)) + data


def decode_attestation_payload_blob(blob: str) -> MinerAttestationPayloadBundle:
    """Decode an opaque attestation payload blob emitted by the native helper."""

    try:
        raw_bytes = base64.b64decode(blob)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise MinerAttestationPayloadDecodeError("payload blob was not valid base64") from exc

    raw = memoryview(raw_bytes)
    if len(raw) < 1:
        raise MinerAttestationPayloadDecodeError("payload blob missing version byte")

    version = raw[0]
    if version == _INNER_VERSION:
        index = 1
        nonce, index = _read_string(raw, index)
        challenge_id, index = _read_string(raw, index)
        heartbeat_nonce, index = _read_string(raw, index)
        artifact, index = _read_string(raw, index)

        if index != len(raw):
            raise MinerAttestationPayloadDecodeError("payload blob contained trailing bytes")

        return MinerAttestationPayloadBundle(
            nonce=nonce,
            challenge_id=challenge_id,
            heartbeat_nonce=heartbeat_nonce,
            artifact=artifact,
            runtime=None,
        )

    if version == _WRAPPED_VERSION:
        expected_len = 1 + _FLOAT64.size + 1 + _UINT32.size
        if len(raw) < expected_len:
            raise MinerAttestationPayloadDecodeError("wrapped payload blob malformed")
        index = 1
        (duration_ms,) = _FLOAT64.unpack_from(raw, index)
        index += _FLOAT64.size
        delay_flag = raw[index]
        index += 1
        inner_blob, index = _read_string(raw, index)
        if index != len(raw):
            raise MinerAttestationPayloadDecodeError("wrapped payload blob contained trailing bytes")

        inner_bundle = decode_attestation_payload_blob(inner_blob)
        inner_bundle.runtime = MinerAttestationRuntime(
            duration_ms=duration_ms,
            delay_suspect=bool(delay_flag),
        )
        return inner_bundle

    raise MinerAttestationPayloadDecodeError(f"unsupported payload blob version {version}")


def wrap_attestation_payload_blob(payload_blob: str, runtime: MinerAttestationRuntime) -> str:
    """Embed runtime data inside a wrapped payload blob."""

    if not isinstance(payload_blob, str):
        raise ValueError("payload_blob must be a base64-encoded string")
    if runtime is None:
        return payload_blob

    raw_parts = [
        bytes([_WRAPPED_VERSION]),
        _FLOAT64.pack(runtime.duration_ms),
        b"\x01" if runtime.delay_suspect else b"\x00",
        _write_string(payload_blob),
    ]

    return base64.b64encode(b"".join(raw_parts)).decode("ascii")


__all__ = [
    "MinerAttestationPayloadBundle",
    "MinerAttestationPayloadDecodeError",
    "decode_attestation_payload_blob",
    "wrap_attestation_payload_blob",
]
