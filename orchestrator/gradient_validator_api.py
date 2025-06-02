from fastapi import APIRouter, HTTPException, Depends, Request, Header
from loguru import logger
from typing import Dict, Optional, Literal, Annotated
import torch
import time
import settings
from pydantic import BaseModel
from orchestrator.serializers import GradientValidationResponse
from gradient_validator.gradient_validator import GradientValidator
from utils.epistula import EpistulaHeaders, create_message_body

router = APIRouter(prefix="/gradient-validator")


class InitializeValidatorRequest(BaseModel):
    layer: int
    miner_hotkey: str
    weight_path: Optional[str] = None


def get_validator(request: Request) -> GradientValidator:
    """
    Get the gradient validator instance from app state.
    """
    if not hasattr(request.app.state, "validator"):
        raise RuntimeError("Validator not initialized in app state")
    return request.app.state.validator


def verify_whitelisted(signed_by: str):
    """Verify that the request is coming from a whitelisted validator."""
    if settings.BITTENSOR:
        if signed_by != settings.ORCHESTRATOR_KEY:
            raise HTTPException(
                status_code=403, detail="Access denied. This endpoint is only accessible to the orchestrator."
            )


@router.post("/initialize", response_model=Dict[str, str])
async def initialize_validator(
    request: InitializeValidatorRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    validator: GradientValidator = Depends(get_validator),
):
    """Initialize the gradient validator to track a specific miner and layer."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body(request.model_dump()), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    verify_whitelisted(signed_by)

    try:
        logger.info(f"Initializing gradient validator for miner {request.miner_hotkey} and layer {request.layer}")
        await validator.reset_validator()

        layer = request.layer
        if layer is None:
            layer = 0  # Default to layer 0 if not specified

        await validator.load_weights(
            layer=layer,
            miner_hotkey=request.miner_hotkey,
            weight_path=request.weight_path,
        )
        return {"message": "Gradient validator initialized successfully"}
    except Exception as e:
        logger.exception(f"Failed to initialize gradient validator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forward", response_model=GradientValidationResponse)
async def forward_activation(
    activation_uid: str,
    direction: Literal["forward", "backward", "initial"],
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    validator: GradientValidator = Depends(get_validator),
):
    """Perform a forward pass with the gradient validator."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body({}), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    verify_whitelisted(signed_by)

    try:
        is_valid, score, reason = await validator.forward(activation_uid=activation_uid, direction=direction)
        # Convert to appropriate types for the response model
        score_value = float(score.item() if isinstance(score, torch.Tensor) else score)
        return GradientValidationResponse(is_valid=bool(is_valid), score=score_value, reason=str(reason))
    except Exception as e:
        logger.error(f"Forward validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backward", response_model=GradientValidationResponse)
async def backward_activation(
    activation_uid: str,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    validator: GradientValidator = Depends(get_validator),
):
    """Perform a backward pass with the gradient validator."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body({}), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    verify_whitelisted(signed_by)

    try:
        is_valid, score, reason = await validator.backward(activation_uid=activation_uid)
        # Convert to appropriate types for the response model
        score_value = float(score.item() if isinstance(score, torch.Tensor) else score)
        return GradientValidationResponse(is_valid=bool(is_valid), score=score_value, reason=str(reason))
    except Exception as e:
        logger.error(f"Backward validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-weights", response_model=GradientValidationResponse)
async def validate_weights(
    weights_path: str,
    metadata_path: str,
    optimizer_state_path: str,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    validator: GradientValidator = Depends(get_validator),
):
    """Validate the weights submitted by a miner."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body({}), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    verify_whitelisted(signed_by)

    try:
        is_valid, score, reason = await validator.validate_weights(
            weights_path=weights_path, metadata_path=metadata_path, optimizer_state_path=optimizer_state_path
        )
        # Convert to appropriate types for the response model
        score_value = float(score.item() if isinstance(score, torch.Tensor) else score)
        return GradientValidationResponse(is_valid=bool(is_valid), score=score_value, reason=str(reason))
    except Exception as e:
        logger.exception(f"Weight validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def health_check(validator: GradientValidator = Depends(get_validator)):
    """Get the current status of the gradient validator."""
    if validator.available:
        return {"status": "healthy", "message": "Gradient validator is not validationg"}
    else:
        return {"status": "healthy", "message": "Gradient validator is validating"}
