import time
from loguru import logger
from typing import Annotated

from utils.bt_utils import verify_entity_type
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

import settings
from orchestrator.orchestrator import orchestrator
from utils.epistula import EpistulaHeaders, create_message_body
from utils.partitions import Partition
from utils.s3_interactions import generate_presigned_url
from storage.activation_storage import ActivationStore
from storage.weight_storage import WeightStore
from storage.serializers import (
    ActivationUploadRequest,
    ActivationDownloadRequest,
    WeightLayerRequest,
    StorageResponse,
    PresignedUrlRequest,
    ActivationResponse,
    MultipartUploadRequest,
    MultipartUploadResponse,
    CompleteMultipartUploadRequest,
)
from uuid import uuid4
from utils.s3_interactions import (
    create_multipart_upload,
    generate_presigned_url_for_part,
    complete_multipart_upload,
    abort_multipart_upload,
)
from utils.shared_states import MergingPhase


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


# Initialize rate limiter
hotkey_limiter = Limiter(key_func=get_signed_by_key)


def get_storage_instances() -> tuple[ActivationStore, WeightStore]:
    """Dependency to get the storage instances from the orchestrator."""
    return orchestrator.activation_store, orchestrator.weight_store


router = APIRouter(prefix="/storage")


# Activation Storage Endpoints
@router.post("/activations/upload", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def upload_activation_to_orchestrator(
    request: Request,  # Required for rate limiting
    activation_request: ActivationUploadRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=activation_request.activation_uid,
        layer=activation_request.layer,
        direction=activation_request.direction,
        miner_hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body(activation_request.model_dump()), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        entity_info = None
        if settings.BITTENSOR:
            # Verify the entity exists in metagraph (accept both miners and validators)
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both miners and validators to upload
            )

            # If it's a miner, verify they're uploading their own activations
            if entity_info["entity_type"] == "miner":
                # Parse the UID from the request if it's numeric
                if (
                    activation_request.activation_uid.isdigit()
                    and int(activation_request.activation_uid) != entity_info["uid"]
                ):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Miner {entity_info['uid']} cannot upload activations for uid {activation_request.activation_uid}",
                    )

        entity_type = entity_info["entity_type"] if entity_info else "unknown"
        entity_uid = entity_info["uid"] if entity_info else "unknown"
        logger.info(
            f"{entity_type.capitalize()} {entity_uid} ({signed_by}) "
            f"uploading activation for activation_uid {activation_request.activation_uid} layer {activation_request.layer}"
        )

        activation_store, _ = storage_instances
        try:
            path = await activation_store.upload_activation_to_activation_store(
                activation_uid=activation_request.activation_uid,
                layer=activation_request.layer,
                direction=activation_request.direction,
                activation_path=activation_request.activation_path,
                miner_hotkey=signed_by,
            )
            return StorageResponse(message="Activation uploaded successfully", data={"path": path})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/activations/download", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def download_activation(
    request: Request,  # Required for rate limiting
    activation_request: ActivationDownloadRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    try:
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body(activation_request.model_dump()), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        entity_info = None
        if activation_request.direction == "forward":
            required_type = "validator"
        elif activation_request.direction == "backward" or activation_request.direction == "initial":
            required_type = None  # Allow both to download
        if settings.BITTENSOR:
            # Verify the entity exists in metagraph (accept both miners and validators)
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=required_type,
            )

        entity_type = entity_info["entity_type"] if entity_info else "unknown"
        entity_uid = entity_info["uid"] if entity_info else "unknown"
        logger.info(
            f"{entity_type.capitalize()} {entity_uid} ({signed_by}) "
            f"downloading activation {activation_request.activation_uid}"
        )
        if settings.BITTENSOR and entity_info["entity_type"] == "miner":
            # check if miner has space in cache
            cached_activations = orchestrator.miner_registry.get_miner_cached_activations(signed_by)
            if activation_request.activation_uid not in cached_activations:
                return StorageResponse(message="Miner does not have activation in cache", data={"path": None})

        activation_store, _ = storage_instances
        try:
            path = await activation_store.download_activation_from_activation_store(
                activation_uid=activation_request.activation_uid,
                direction=activation_request.direction,
                layer=activation_request.layer,
                delete=activation_request.delete,
                miner_hotkey=signed_by,
                fetch_historic=activation_request.fetch_historic,
            )

            return StorageResponse(message="Activation downloaded successfully", data={"path": path})
        except Exception as e:
            logger.error(f"Error downloading activation: {e}")
            return StorageResponse(message="Error downloading activation", data={"path": None})
    except Exception as e:
        logger.error(f"Error downloading activation: {e}")
        return StorageResponse(message="Error downloading activation", data={"path": None})


@router.post("/activations/random", response_model=ActivationResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def get_random_activation(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    """Get a random activation without exposing the full list to the miner."""
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        orchestrator.validate_state(expected_status=MergingPhase.IS_TRAINING, hotkey=signed_by)
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

        entity_info = None
        if settings.BITTENSOR:
            # Verify the entity exists in metagraph
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both to get random activation
            )

        entity_type = entity_info["entity_type"] if entity_info else "unknown"
        entity_uid = entity_info["uid"] if entity_info else "unknown"
        logger.info(f"{entity_type.capitalize()} {entity_uid} ({signed_by}) " f"requesting activation")
        # check if miner has space in cache
        try:
            activation_response = await orchestrator.get_miner_activation(hotkey=signed_by)
            return activation_response
        except ValueError as e:
            logger.error(f"Error getting miner activation: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception(f"Error getting miner activation: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/activations/stats", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def get_activation_stats(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
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

        entity_info = None
        if settings.BITTENSOR:
            # Verify the entity exists in metagraph
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both to get stats
            )

        entity_type = entity_info["entity_type"] if entity_info else "unknown"
        entity_uid = entity_info["uid"] if entity_info else "unknown"
        logger.info(f"{entity_type.capitalize()} {entity_uid} ({signed_by}) " f"getting activation stats")

        activation_store, _ = storage_instances
        try:
            stats = await activation_store.get_activations_stats()
            return StorageResponse(message="Activation stats retrieved successfully", data={"stats": stats})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/activations/is_active")
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def is_activation_active(
    request: Request,  # Required for rate limiting
    layer: int,
    activation_uid: str,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
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

        if settings.BITTENSOR:
            # Verify the entity exists in metagraph
            verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both to check
            )

        activation_store, _ = storage_instances
        try:
            is_active = await activation_store.is_activation_active(layer=layer, activation_uid=activation_uid)
            return {"is_active": is_active}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights/{miner_hotkey}", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def get_weights(
    request: Request,  # Required for rate limiting
    miner_hotkey: str,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
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

        if settings.BITTENSOR:
            # Verify the entity exists in metagraph (allow both miners and validators to get weights)
            verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both
            )

        logger.info(f"Miner {miner_hotkey[:8]}... getting weights")

        _, weight_store = storage_instances
        try:
            weights = await weight_store.get_miner_weights(miner_hotkey=miner_hotkey)
            return StorageResponse(message="Weights retrieved successfully", data={"weights": weights})
        except KeyError:
            raise HTTPException(status_code=404, detail="Weights not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights/list", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def list_miners(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
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

        entity_info = None
        if settings.BITTENSOR:
            # Verify the entity exists in metagraph
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both
            )

        entity_type = entity_info["entity_type"] if entity_info else "unknown"
        entity_uid = entity_info["uid"] if entity_info else "unknown"
        logger.info(f"{entity_type.capitalize()} {entity_uid} ({signed_by}) " f"listing miners with weights")

        _, weight_store = storage_instances
        try:
            miners = await weight_store.list_miners()
            return StorageResponse(message="Miners listed successfully", data={"miners": miners})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.delete("/weights", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def delete_weights(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        validator_hotkey = signed_by
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

        if settings.BITTENSOR:
            # Verify the entity is a validator (only validators should delete weights)
            verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type="validator",
            )

        logger.info(f"Validator({signed_by[:8]}) deleting weights for miner {validator_hotkey[:8]}...")

        _, weight_store = storage_instances
        try:
            await weight_store.delete_weights(miner_hotkey=validator_hotkey)
            return StorageResponse(message="Weights deleted successfully")
        except KeyError:
            raise HTTPException(status_code=404, detail="Weights not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/weights/set_layer_weights", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def set_layer_weights(
    request: Request,  # Required for rate limiting
    weight_request: WeightLayerRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body(weight_request.model_dump()), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        entity_info = None
        if settings.BITTENSOR:
            # Verify the entity is a validator (only validators should set layer weights)
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type="validator",
            )

        entity_uid = entity_info["uid"] if entity_info else "unknown"
        logger.info(f"Validator {entity_uid} ({signed_by}) setting layer {weight_request.layer} weights")

        _, weight_store = storage_instances
        try:
            await weight_store.set_layer_weights(
                layer=weight_request.layer,
                weights=weight_request.weights,
            )
            return StorageResponse(message="Layer weights set successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights/layer/{layer}", response_model=list[Partition])
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def get_layer_weights(
    request: Request,  # Required for rate limiting
    layer: int,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    storage_instances: tuple[ActivationStore, WeightStore] = Depends(get_storage_instances),
) -> list[Partition]:
    with logger.contextualize(
        activation_uid=None,
        layer=layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        try:
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

            if settings.BITTENSOR:
                # Verify the entity exists in metagraph
                verify_entity_type(
                    signed_by=signed_by,
                    metagraph=orchestrator.metagraph,
                    required_type=None,  # Allow both
                )

            logger.info(f"Getting layer {layer} weights, asked for by {signed_by[:8]}...")

            _, weight_store = storage_instances
            try:
                layer_partitions = await weight_store.get_layer_partitions(layer=layer)
                logger.debug(f"Layer {layer} partitions: {layer_partitions}")
                return layer_partitions
            except KeyError:
                raise HTTPException(status_code=404, detail="Layer weights not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        except Exception as ex:
            logger.error(f"Failed to get layer weights: {ex}")


@router.get("/presigned_url", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def get_presigned_url(
    request: Request,  # Required for rate limiting
    data: PresignedUrlRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    """Generate a presigned URL for S3 operations.

    Args:
        path: The S3 key of the object
        method: The S3 operation to allow ('get_object' or 'put_object')
        expires_in: Number of seconds until the URL expires

    Returns:
        PresignedUrlResponse containing the URL and any additional fields
    """
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body(data.model_dump()), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        entity_info = None
        if settings.BITTENSOR:
            # Verify the entity exists in metagraph
            entity_info = verify_entity_type(
                signed_by=signed_by,
                metagraph=orchestrator.metagraph,
                required_type=None,  # Allow both
            )

            entity_type = entity_info["entity_type"] if entity_info else "unknown"
            entity_uid = entity_info["uid"] if entity_info else "unknown"
            logger.info(
                f"{entity_type.capitalize()} {entity_uid} ({signed_by}) " f"requesting presigned URL for {data.path}"
            )
        else:
            entity_uid = 0  # TODO: What should the uid be in Mock mode?

        try:
            presigned_data = generate_presigned_url(
                path=data.path,
                expires_in=data.expires_in,
            )

            return StorageResponse(
                message="Presigned URL generated successfully",
                data={"presigned_data": presigned_data},
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/multipart_upload/initiate", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def initiate_multipart_upload(
    request: Request,  # Required for rate limiting
    upload_request: MultipartUploadRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    """Initiate a multipart upload and return presigned URLs for parts."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body(upload_request.model_dump()), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    entity_info = None
    if settings.BITTENSOR:
        # Verify the entity exists in metagraph
        entity_info = verify_entity_type(
            signed_by=signed_by,
            metagraph=orchestrator.metagraph,
            required_type=None,  # Allow both miners and validators
        )

    entity_type = entity_info["entity_type"] if entity_info else "unknown"
    entity_uid = entity_info["uid"] if entity_info else "unknown"
    logger.info(
        f"{entity_type.capitalize()} {entity_uid} ({signed_by}) "
        f"initiating multipart upload for {upload_request.path} ({upload_request.file_size} bytes)"
    )

    try:
        # Check if multipart upload is actually needed
        min_part_size = 5 * 1024 * 1024  # 5MB minimum per AWS
        max_single_part_size = 5 * 1024 * 1024 * 1024  # 5GB maximum for single part

        if upload_request.file_size <= max_single_part_size and upload_request.file_size <= upload_request.part_size:
            # Use regular single-part upload
            presigned_data = generate_presigned_url(
                path=upload_request.path,
                expires_in=upload_request.expires_in,
            )

            return StorageResponse(
                message="Single-part upload URL generated",
                data={"use_multipart": False, "presigned_data": presigned_data},
            )

        # Use multipart upload
        upload_id = create_multipart_upload(upload_request.path)

        # Calculate parts needed
        actual_part_size = max(upload_request.part_size, min_part_size)
        total_parts = (upload_request.file_size + actual_part_size - 1) // actual_part_size

        # Generate presigned URLs for each part
        presigned_urls = []
        part_numbers = []

        for part_number in range(1, total_parts + 1):
            presigned_url = generate_presigned_url_for_part(
                path=upload_request.path,
                upload_id=upload_id,
                part_number=part_number,
                expires_in=upload_request.expires_in,
            )
            presigned_urls.append({"url": presigned_url})
            part_numbers.append(part_number)

        response_data = MultipartUploadResponse(
            upload_id=upload_id, presigned_urls=presigned_urls, part_numbers=part_numbers
        )

        return StorageResponse(
            message="Multipart upload initiated successfully",
            data={
                "use_multipart": True,
                "multipart_data": response_data.model_dump(),
                "part_size": actual_part_size,
                "total_parts": total_parts,
            },
        )

    except Exception as e:
        logger.error(f"Error initiating multipart upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multipart_upload/complete", response_model=StorageResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def complete_multipart_upload_endpoint(
    request: Request,  # Required for rate limiting
    complete_request: CompleteMultipartUploadRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    """Complete a multipart upload."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body(complete_request.model_dump()), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    if settings.BITTENSOR:
        # Verify the entity exists in metagraph
        verify_entity_type(
            signed_by=signed_by,
            metagraph=orchestrator.metagraph,
            required_type=None,  # Allow both
        )

    logger.info(f"Completing multipart upload {complete_request.upload_id} for {complete_request.path}")

    try:
        s3_path = complete_multipart_upload(
            path=complete_request.path, upload_id=complete_request.upload_id, parts=complete_request.parts
        )

        return StorageResponse(message="Multipart upload completed successfully", data={"s3_path": s3_path})

    except Exception as e:
        logger.error(f"Error completing multipart upload: {e}")
        # Try to abort the upload on failure
        try:
            abort_multipart_upload(complete_request.path, complete_request.upload_id)
        except Exception:
            pass  # Ignore errors during abort
        raise HTTPException(status_code=500, detail=str(e))
