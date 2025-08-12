import torch
from subnet.utils.vector_utils import check_for_nans_and_infs
from common.utils.exceptions import NanInfWarning


def tensor_checks(validator_flat: torch.Tensor, miner_flat: torch.Tensor) -> None:
    """Function to handle tensor based checks for the validator and miner tensors.

    Args:
        validator_flat (torch.Tensor): The validator tensor.
        miner_flat (torch.Tensor): The miner tensor.
    """

    assert validator_flat.shape == miner_flat.shape, "Validator and miner tensors must have the same shape"
    assert (
        validator_flat.device == miner_flat.device
    ), f"Tensors must be on the same device, got validator tensor on {validator_flat.device} and miner tensor on {miner_flat.device}"

    check_for_nans_and_infs(validator_flat, "validator_flat", exception_type=NanInfWarning)
    check_for_nans_and_infs(miner_flat, "miner_flat", exception_type=NanInfWarning)


def compute_cosine_similarity(validator_flat: torch.Tensor, miner_flat: torch.Tensor) -> float:
    """Compute the cosine similarity between the validator and miner tensors.

    Args:
        validator_flat (torch.Tensor): The validator tensor.
        miner_flat (torch.Tensor): The miner tensor.

    Returns:
        float: The cosine similarity.
    """
    tensor_checks(validator_flat=validator_flat, miner_flat=miner_flat)

    validator_flat = validator_flat.unsqueeze(0)
    miner_flat = miner_flat.unsqueeze(0)
    similarity: torch.Tensor = torch.cosine_similarity(validator_flat, miner_flat, dim=1)
    score = similarity.item() if similarity.numel() == 1 else similarity[0].item()

    return score


def compute_magnitude_ratio(validator_flat: torch.Tensor, miner_flat: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute the magnitude ratio between the validator and miner tensors.

    R(v, m) = min(||v||, ||m||) / (max(||v||, ||m||) + ε)

    Args:
        validator_flat (torch.Tensor): The validator tensor.
        miner_flat (torch.Tensor): The miner tensor.
        eps (float): The epsilon value to avoid division by zero.

    Returns:
        float: The magnitude ratio.
    """

    tensor_checks(validator_flat=validator_flat, miner_flat=miner_flat)

    validator_norm = torch.norm(validator_flat).item()
    miner_norm = torch.norm(miner_flat).item()

    magnitude_ratio = torch.min(torch.tensor(validator_norm), torch.tensor(miner_norm)) / (
        torch.max(torch.tensor(validator_norm), torch.tensor(miner_norm)) + eps
    )

    return magnitude_ratio.item()


def apply_burn_factor(raw_weights: torch.Tensor, burn_factor: float, netuid: int, owner_uid: int = 209) -> torch.Tensor:
    """Apply the burn factor to the raw weights.

    Args:
        raw_weights (torch.Tensor): The raw weights.
        burn_factor (float): The burn factor.
        netuid (int): The netuid.
        owner_uid (int): The owner uid.
    """
    if burn_factor <= 1 and netuid == 9 and raw_weights[owner_uid] > burn_factor:
        raw_weights = raw_weights * (1 - burn_factor)

        # Add the 1-1/burn factor to the owner uid
        if len(raw_weights) > owner_uid:
            raw_weights[owner_uid] = burn_factor

    return raw_weights
