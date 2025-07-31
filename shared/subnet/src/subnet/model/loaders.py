from loguru import logger
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from typing import Union, List

from subnet.model.llama3.splits import (
    Llama3HeadModel,
    Llama3TailModel,
    Llama3BodyModel,
)
from subnet.model.llama3.full import Llama3Model, Llama3ModelNativeBottlenecks


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
    torch.cuda.manual_seed_all(seed)
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


def _sample_packing_generator(dataset: Dataset, tokenizer: PreTrainedTokenizer, batch_size: int, max_length: int):
    """
    Generator function to yield packed samples
    """
    buffer = []
    for sample_idx, sample in enumerate(dataset):
        # Get the sample rank (in the logic of the agent rank)
        """This enables different miners to load different samples

        if sample_rank != self.trainer_client.stage_rank:
            continue
        """
        # Tokenize the text
        tokens = tokenizer(sample["text"], truncation=False, add_special_tokens=False)["input_ids"]

        # Add tokens to buffer
        buffer.extend(tokens)
        buffer.append(tokenizer.eos_token_id)

        # Yield packed samples if buffer length reaches or exceeds target length
        while len(buffer) >= batch_size * max_length:
            packed_sample = buffer[: batch_size * max_length]  # Get packed sample of target length
            packed_sample_tensor = torch.tensor(packed_sample).view(
                batch_size, max_length
            )  # Convert to torch tensor and reshape
            yield packed_sample_tensor  # Yield the packed sample tensor
            buffer = buffer[batch_size * max_length :]  # Remove used tokens from buffer


def _sample_unpacking_generator(dataset: Dataset, tokenizer: PreTrainedTokenizer, batch_size: int, max_length: int):
    """
    Generator function to yield unpacked samples
    """
    buffer = []
    for sample_idx, sample in enumerate(dataset):
        """This enables different miners to load different samples
        # Get the sample rank (in the logic of the agent rank)
        sample_rank = sample_idx % self.trainer_client.stage_size

        if sample_rank != self.trainer_client.stage_rank:
            continue
        """

        # Tokenize the text
        tokens = tokenizer(sample["text"], truncation=True, add_special_tokens=False)["input_ids"]

        tokens = tokens[:max_length]
        tokens += [tokenizer.eos_token_id] * (max_length - len(tokens))

        # Add tokens to buffer
        buffer.extend(tokens)

        # Yield packed samples if buffer length reaches or exceeds target length
        while len(buffer) >= batch_size * max_length:
            packed_sample = buffer[: batch_size * max_length]  # Get packed sample of target length
            packed_sample_tensor = torch.tensor(packed_sample).view(
                batch_size, max_length
            )  # Convert to torch tensor and reshape
            yield packed_sample_tensor  # Yield the packed sample tensor
            buffer = buffer[batch_size * max_length :]  # Remove used tokens from buffer


def load_dataloader(
    hf_token: str,
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    sequence_length: int,
    pack_samples: bool = True,
    shuffle_dataset: bool = True,
):
    """
    Load the dataloader for the dataset.
    """
    dataset = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
    # shuffle the dataset
    if shuffle_dataset:
        dataset = dataset.shuffle()

    # Choose the appropriate data generator based on args
    if pack_samples:
        logger.info("Using sample packing generator")
        dataloader = _sample_packing_generator(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=sequence_length,
        )
    else:
        logger.info("Using sample packing generator")
        dataloader = _sample_unpacking_generator(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=sequence_length,
        )
    if dataloader is None:
        raise ValueError("Dataloader is None")
    return dataloader


if __name__ == "__main__":
    from common.configs import LLAMA32_CONFIG_1B

    model = load_model_split(
        model_cfg=LLAMA32_CONFIG_1B,
        model_split=[[-1, 5], [5, 10], [10, -1]],
        device="cuda",
        seed=42,
    )
    print(model)
