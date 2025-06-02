from typing import Literal, Optional, Any
from pydantic import BaseModel, ConfigDict
import torch


class ActivationUploadRequest(BaseModel):
    activation_uid: str
    layer: int
    direction: Literal["forward", "backward", "initial"]
    activation_path: str


class ActivationPresignedUrlRequest(BaseModel):
    activation_uid: str
    layer: int
    direction: Literal["forward", "backward", "initial"]
    activation_path: str


class ActivationDownloadRequest(BaseModel):
    activation_uid: str
    direction: Literal["forward", "backward", "initial"]
    layer: int | None = None
    delete: bool = True
    fetch_historic: bool = False


class ActivationListRequest(BaseModel):
    layer: int
    direction: Literal["forward", "backward", "initial"]
    include_pending: bool = False


class ActivationRandomRequest(BaseModel):
    layer: int
    direction: Literal["forward", "backward", "initial"]


class WeightUploadRequest(BaseModel):
    weights: torch.Tensor
    partition_dict: dict
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WeightLayerRequest(BaseModel):
    layer: int
    weights: torch.Tensor
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ActivationResponse(BaseModel):
    activation_uid: str | None = None
    activation_path: str | None = None
    direction: Literal["forward", "backward", "initial"] | None = None
    initial_activation: str | None = None
    initial_activation_path: str | None = None
    reason: str | None = None


class StorageResponse(BaseModel):
    message: str
    data: Optional[dict] = None


class PresignedUrlRequest(BaseModel):
    path: str
    expires_in: int = 3600


class MultipartUploadRequest(BaseModel):
    path: str
    file_size: int
    part_size: int = 100 * 1024 * 1024  # 100MB default
    expires_in: int = 3600


class MultipartUploadResponse(BaseModel):
    upload_id: str
    presigned_urls: list[dict[str, str]]  # List of presigned URLs for each part
    part_numbers: list[int]  # Corresponding part numbers


class CompleteMultipartUploadRequest(BaseModel):
    path: str
    upload_id: str
    parts: list[dict[str, Any]]  # List of {PartNumber: int, ETag: str}


class AbortMultipartUploadRequest(BaseModel):
    path: str
    upload_id: str
