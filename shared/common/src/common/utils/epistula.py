import orjson
import time
from hashlib import sha256
from math import ceil
import traceback
from typing import Any, Optional
from fastapi import HTTPException, Header
from loguru import logger
from pydantic import BaseModel, model_validator
from substrateinterface import Keypair
import uuid
from common import settings as common_settings

HEADER_REQUEST_ID = "X-Request-Id"


class EpistulaError(BaseModel):
    error: str
    message: Optional[str] = None
    name: Optional[str] = None

    @model_validator(mode="after")
    def make_message(self):
        self.name = self.__class__.__name__
        self.message = f"{self.name}: {self.error}"
        return self


class EpistulaHeaders(BaseModel):
    timestamp: str = Header(default=str(time.time()), alias="Epistula-Timestamp")
    signed_by: str = Header(..., alias="Epistula-Signed-By")
    request_signature: str = Header(..., alias="Epistula-Request-Signature")
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias=HEADER_REQUEST_ID)
    spec_version: str = Header(default=int(0), alias="X-Spec-Version")

    def verify_signature_v2(self, body: bytes, now: float, timeout: int):
        """
        Verify the signature of the request.

        Args:
            body: The body of the request
            now: The current time
            timeout: The timeout in milliseconds

        Returns:
            None if the signature is valid, otherwise an error message
        """
        try:
            if not isinstance(self.request_signature, str):
                raise ValueError("Invalid Signature")

            timestamp = int(float(self.timestamp))
            if not isinstance(timestamp, int):
                raise ValueError("Invalid Timestamp")

            if not isinstance(self.signed_by, str):
                raise ValueError("Invalid Sender key")

            if not isinstance(body, bytes):
                raise ValueError("Body is not of type bytes")

            keypair = Keypair(ss58_address=self.signed_by)

            if timestamp + timeout < now:
                raise ValueError("Request is too stale")
            message = f"{sha256(body).hexdigest()}.{self.timestamp}."

            verified = keypair.verify(message, self.request_signature)
            if not verified:
                raise ValueError("Signature Mismatch")

        except ValueError as e:
            logger.error("signature_verification_failed", error=e)
            raise HTTPException(status_code=400, detail=EpistulaError(error=str(e)).model_dump())
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

    # Create message for signing with optional signed_for
    message = f"{sha256(body).hexdigest()}.{timestamp}.{signed_for or ''}"

    headers = {
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x" + hotkey.sign(message).hex(),
        "X-Spec-Version": str(common_settings.__SPEC_VERSION__),
        HEADER_REQUEST_ID: str(uuid.uuid4()),
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
    return orjson.dumps(data, default=str, option=orjson.OPT_SORT_KEYS)
