import torch
from torch.optim import Optimizer

import settings


def add_artificial_gradients(model):
    model.to(settings.DEVICE)
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param.data).to(dtype=torch.bfloat16).to(settings.DEVICE)


def flatten_optimizer_state(optimizer, dtype=torch.bfloat16):
    """Flatten all tensors in optimizer state dict into a single tensor."""
    state_dict = optimizer.state_dict()
    tensors = []
    tensor_shapes = []

    if len(state_dict["state"]) == 0:
        optimizer.step()

    for group in state_dict["state"].values():
        for k, v in group.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                tensors.append(v.flatten().to(dtype).to(settings.DEVICE))
                tensor_shapes.append(v.shape)

    flat_tensor = torch.cat(tensors)
    return flat_tensor, tensor_shapes, state_dict


def reconstruct_optimizer_state(flat_tensor, tensor_shapes, state_dict, optimizer: Optimizer):
    """Reconstruct optimizer state dict from flattened tensor."""
    new_state_dict = state_dict.copy()

    start_idx = 0
    tensor_idx = 0
    for group in new_state_dict["state"].values():
        for k, v in group.items():
            if k == "step":
                continue
            if isinstance(v, torch.Tensor):
                numel = v.numel()
                tensor_data = flat_tensor[start_idx : start_idx + numel]
                group[k] = tensor_data.reshape(tensor_shapes[tensor_idx])
                start_idx += numel
                tensor_idx += 1

    optimizer.load_state_dict(new_state_dict)
    return optimizer


def check_for_nans(tensor, name: str | None = None):
    # Check to see if the weights or optimizer state have any nans
    tensor = tensor.to("cpu")
    try:
        num_nans = torch.isnan(tensor).sum()
        if num_nans > 0:
            total = tensor.numel()
            percentage = (num_nans / total) * 100
            logger.error(f"❌ Miner has NaNs in {name} | {num_nans} / {total} = {percentage:.2f}%")
            raise Exception(f"{name} has NaNs")
    except Exception as e:
        raise e
