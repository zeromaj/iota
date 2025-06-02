import json
import time
from hashlib import sha256
from math import ceil
import traceback
from typing import Annotated, Any, Optional
from uuid import uuid4

from fastapi import Header
from loguru import logger
from pydantic import BaseModel
from substrateinterface import Keypair

from settings import SIGNATURE_TIMEOUT_MS


class EpistulaHeaders(BaseModel):
    version: str = Header(..., alias="Epistula-Version")
    timestamp: str = Header(default=str(time.time()), alias="Epistula-Timestamp")
    uuid: str = Header(..., alias="Epistula-Uuid")
    signed_by: str = Header(..., alias="Epistula-Signed-By")
    request_signature: str = Header(..., alias="Epistula-Request-Signature")

    def verify_signature_v2(self, body: bytes, now: float) -> Optional[Annotated[str, "Error Message"]]:
        try:
            if not isinstance(self.request_signature, str):
                raise ValueError("Invalid Signature")

            timestamp = int(float(self.timestamp))
            if not isinstance(timestamp, int):
                raise ValueError("Invalid Timestamp")

            if not isinstance(self.signed_by, str):
                raise ValueError("Invalid Sender key")

            if not isinstance(self.uuid, str):
                raise ValueError("Invalid uuid")

            if not isinstance(body, bytes):
                raise ValueError("Body is not of type bytes")

            keypair = Keypair(ss58_address=self.signed_by)

            if timestamp + SIGNATURE_TIMEOUT_MS < now:
                raise ValueError("Request is too stale")

            message = f"{sha256(body).hexdigest()}.{self.uuid}.{self.timestamp}."

            verified = keypair.verify(message, self.request_signature)
            if not verified:
                raise ValueError("Signature Mismatch")

            return None

        except Exception as e:
            logger.error("signature_verification_failed", error=traceback.format_exc())
            return str(e)


def generate_header(
    hotkey: Keypair,
    body: bytes,
    signed_for: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate headers containing signatures and metadata for a message.

    Args:
        hotkey: The keypair used for signing
        body: The message body in bytes
        signed_for: Receiver's address (optional)

    Returns:
        Dictionary containing all necessary headers
    """
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())

    # Create message for signing with optional signed_for
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}"

    headers = {
        "Epistula-Version": "2",
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x" + hotkey.sign(message).hex(),
    }

    # Only add signed_for related headers if it's specified
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        # Generate time-based signatures for the interval
        for i, interval_offset in enumerate([-1, 0, 1]):
            signature = "0x" + hotkey.sign(f"{timestampInterval + interval_offset}.{signed_for}").hex()
            headers[f"Epistula-Secret-Signature-{i}"] = signature

    return headers


def create_message_body(data: dict) -> bytes:
    """Utility method to create message body from dictionary data"""
    return json.dumps(data, default=str, sort_keys=True).encode("utf-8")
