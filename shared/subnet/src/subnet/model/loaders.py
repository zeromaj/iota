from loguru import logger
import torch
import random
import numpy as np
from typing import Union, List

from subnet.model.llama3.splits import (
    Llama3HeadModel,
    Llama3TailModel,
    Llama3BodyModel,
)
from subnet.model.llama3.full import Llama3Model, Llama3ModelNativeBottlenecks
from subnet.model import gpu_device


def load_model_split(
    model_cfg: dict,
    model_split: Union[list[int], List[List[int]]],
    device: Union[str, torch.device],
    seed: int = 42,
):
    """
    Load the model and split it according to the model_split parameter.
    The model_split parameter is a list of two integers:
    - If [-1, layer_num], take from the beginning of the model until layer_num
    - If [layer_num, -1], take from layer_num until the end
    - If [layer_num1, layer_num2], take from layer_num1 to layer_num2
    - If [-1, -1], use the full model

    Args:
        model_name: Name of the model to load
        model_split: List of two integers indicating the layer split
        device: Device to load the model on ('cpu', 'cuda', etc)
        seed: Random seed for deterministic initialization
    """
    # Set random seeds for deterministic initialization
    torch.manual_seed(seed)
    gpu_device.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if the model split is a list of lists, then one model is split into multiple parts
    # with bottleneck layers
    if isinstance(model_split, list) and isinstance(model_split[0], list):
        bottleneck_positions = [end_layer[1] - 1 for end_layer in model_split[:-1]]

        model = Llama3ModelNativeBottlenecks(cfg=model_cfg, bottleneck_positions=bottleneck_positions, device=device)
        return model

    # if the model split is a list of ints, then start and end layers
    # are defined for splitting
    elif isinstance(model_split, list) and isinstance(model_split[0], int):
        start_layer, end_layer = model_split
    else:
        raise ValueError(f"Invalid model split configuration: {model_split}.")

    # Special case: [-1, -1] means use the full model
    if start_layer == -1 and end_layer == -1:
        model = Llama3Model(cfg=model_cfg, device=device)

    # Case 1: [-1, layer_num] - Head model (from beginning to layer_num)
    elif start_layer == -1 and end_layer >= 0:
        model = Llama3HeadModel(cfg=model_cfg, end_layer=end_layer, device=device)

    # Case 2: [layer_num, -1] - Tail model (from layer_num to end)
    elif start_layer >= 0 and end_layer == -1:
        model = Llama3TailModel(cfg=model_cfg, start_layer=start_layer, device=device)

    # Case 3: [layer_num1, layer_num2] - Body model (from layer_num1 to layer_num2)
    elif start_layer >= 0 and end_layer >= 0:
        model = Llama3BodyModel(
            cfg=model_cfg,
            start_layer=start_layer,
            end_layer=end_layer,
            device=device,
        )

    else:
        raise ValueError(f"Invalid model split configuration: {model_split}.")

    logger.info(f"Model loaded successfully: {model}")
    return model


if __name__ == "__main__":
    from common.configs import LLAMA32_CONFIG_1B

    model = load_model_split(
        model_cfg=LLAMA32_CONFIG_1B,
        model_split=[[-1, 5], [5, 10], [10, -1]],
        device="cuda",
        seed=42,
    )
    print(model)
