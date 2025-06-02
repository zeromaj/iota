# The llama 3 implementation was adapted from Sebastian Raschka's one.

# The original implementation is available at:
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

import torch
import torch.nn as nn


from model.llama3.modules import (
    TransformerBlock,
    compute_rope_params,
    rescale_theta,
    init_weights,
    BottleneckTransformerBlock,
)
from model.utils import quantize_activations_uint8_uniform


class Llama3HeadModel(nn.Module):
    def __init__(self, cfg, end_layer: int, device: str = "cpu"):
        super().__init__()
        self.device = device

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = (
            nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
                [TransformerBlock(cfg) for _ in range(end_layer - 1)]
            )
        )

        # Bottleneck transformer block for the last layer if bottleneck_dim is not None
        if cfg["bottleneck_dim"] is not None:
            self.trf_blocks.append(BottleneckTransformerBlock(cfg))
        else:
            self.trf_blocks.append(TransformerBlock(cfg))

        # Reusuable utilities
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool(),
            persistent=False,
        )

        if cfg["orig_context_length"] != cfg["context_length"]:
            cfg["rope_base"] = rescale_theta(cfg["rope_base"], cfg["orig_context_length"], cfg["context_length"])
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"],
        )

        # apply initialization weights
        self.apply(init_weights)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.to(device)

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)

        state = {}

        if self.cfg["quantize_activations"]:
            state["activations"] = x
            x = quantize_activations_uint8_uniform(x.detach())

        return x, state

    def backward(
        self,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        # If the activations are quantized, we need to use original activations for the backward pass
        if self.cfg["quantize_activations"]:
            output_activations = state["activations"]

        output_activations.backward(activation_grads)


class Llama3BodyModel(nn.Module):
    def __init__(self, cfg, start_layer: int, end_layer: int, device: str = "cpu"):
        super().__init__()
        self.device = device

        self.trf_blocks = (
            nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
                [TransformerBlock(cfg) for _ in range(end_layer - start_layer - 1)]
            )
        )

        # Bottleneck transformer block for the last layer if bottleneck_dim is not None
        if cfg["bottleneck_dim"] is not None:
            self.trf_blocks.append(BottleneckTransformerBlock(cfg))
        else:
            self.trf_blocks.append(TransformerBlock(cfg))

        # Reusuable utilities
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool(),
            persistent=False,
        )

        if cfg["orig_context_length"] != cfg["context_length"]:
            cfg["rope_base"] = rescale_theta(cfg["rope_base"], cfg["orig_context_length"], cfg["context_length"])
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"],
        )

        # apply initialization weights
        self.apply(init_weights)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.to(device)

    def forward(self, x):
        # Cast in case the input is not in cfg["dtype"]
        x = x.to(self.cfg["dtype"])

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)

        state = {}

        if self.cfg["quantize_activations"]:
            state["activations"] = x
            x = quantize_activations_uint8_uniform(x.detach())

        return x, state

    def backward(
        self,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        # If the activations are quantized, we need to use original activations for the backward pass
        if self.cfg["quantize_activations"]:
            output_activations = state["activations"]

        output_activations.backward(activation_grads)


class Llama3TailModel(nn.Module):
    def __init__(self, cfg, start_layer: int, device: str = "cpu", **kwargs):
        super().__init__()

        self.trf_blocks = (
            nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
                [TransformerBlock(cfg) for _ in range(cfg["n_layers"] - start_layer)]
            )
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool(),
            persistent=False,
        )

        if cfg["orig_context_length"] != cfg["context_length"]:
            cfg["rope_base"] = rescale_theta(cfg["rope_base"], cfg["orig_context_length"], cfg["context_length"])
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"],
        )

        # apply initialization weights
        self.apply(init_weights)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.to(device)

    def forward(self, x):
        # Cast in case the input is not in cfg["dtype"]
        x = x.to(self.cfg["dtype"])

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)

        # RMSNorm casts bf16 to bf32, that's why we need to cast back to bf16
        # before the final linear layer
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        state = {}
        return logits, state

    def backward(
        self,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        output_activations.backward(activation_grads)
