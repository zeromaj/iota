# The llama 3 implementation was adapted from Sebastian Raschka's one.

# The original implementation is available at:
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

from typing import Union
import torch
import torch.nn as nn

from subnet.model.llama3.modules import (
    TransformerBlock,
    BottleneckTransformerBlock,
    compute_rope_params,
    rescale_theta,
    BottleneckEncoder,
    BottleneckDecoder,
    init_weights,
)
from subnet.model.utils import convert_dtype_string


class Llama3Model(nn.Module):
    def __init__(self, cfg, device: Union[str, torch.device] = "cpu", **kwargs):
        super().__init__()

        # Main model parameters
        cfg["dtype"] = convert_dtype_string(cfg["dtype"])
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = (
            nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
                [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
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
        ## apply initialization weights
        self.apply(init_weights)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.to(device)

    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)

        state = {}

        # RMSNorm casts bf16 to bf32, that's why we need to cast back to bf16
        # before the final linear layer
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits, state


class Llama3ModelWithBottlenecks(nn.Module):
    def __init__(self, cfg, bottleneck_positions: list[int] = [], device: str = "cpu", **kwargs):
        super().__init__()

        self.bottleneck_positions = bottleneck_positions

        # Main model parameters
        cfg["dtype"] = convert_dtype_string(cfg["dtype"])
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # Bottleneck encoders
        self.bottleneck_encoders = torch.nn.ModuleList(
            [
                (
                    BottleneckEncoder(cfg["bottleneck_dim"], cfg["emb_dim"], cfg["dtype"])
                    if cfg["bottleneck_dim"] is not None
                    else nn.Identity()
                )
                for _ in range(len(bottleneck_positions))
            ]
        )

        # Bottleneck decoders
        self.bottleneck_decoders = torch.nn.ModuleList(
            [
                (
                    BottleneckDecoder(cfg["bottleneck_dim"], cfg["emb_dim"], cfg["dtype"])
                    if cfg["bottleneck_dim"] is not None
                    else nn.Identity()
                )
                for _ in range(len(bottleneck_positions))
            ]
        )

        self.trf_blocks = (
            nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
                [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
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

        ## apply initialization weights
        self.apply(init_weights)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.to(device)

    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        x_bottlenecks = []

        for i, block in enumerate(self.trf_blocks):
            x = block(x, self.mask, self.cos, self.sin)

            if i in self.bottleneck_positions:
                bottleneck_idx = self.bottleneck_positions.index(i)
                x = self.bottleneck_encoders[bottleneck_idx](x)
                x_bottlenecks.append(x)
                x = self.bottleneck_decoders[bottleneck_idx](x)

        state = {"bottlenecks": x_bottlenecks}

        # RMSNorm casts bf16 to bf32, that's why we need to cast back to bf16
        # before the final linear layer
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits, state


class Llama3ModelNativeBottlenecks(nn.Module):
    def __init__(self, cfg, bottleneck_positions: list[int] = [], device: str = "cpu", **kwargs):
        super().__init__()

        self.bottleneck_positions = bottleneck_positions

        # Main model parameters
        cfg["dtype"] = convert_dtype_string(cfg["dtype"])
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        """
        self.bottleneck_trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [BottleneckTransformerBlock(cfg) for _ in range(len(bottleneck_positions))]
        )

        # normal transformer blocks
        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"] - len(bottleneck_positions))]
        )
        """

        # a mix of bottleneck and normal transformer blocks
        self.trf_blocks = nn.ModuleList(
            [
                BottleneckTransformerBlock(cfg) if i in bottleneck_positions else TransformerBlock(cfg)
                for i in range(cfg["n_layers"])
            ]
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

        ## apply initialization weights
        self.apply(init_weights)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.to(device)

    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        x_bottlenecks = []

        for i, block in enumerate(self.trf_blocks):
            x = block(x, self.mask, self.cos, self.sin)
            if i in self.bottleneck_positions:
                x_bottlenecks.append(x)

        state = {"bottlenecks": x_bottlenecks}

        # RMSNorm casts bf16 to bf32, that's why we need to cast back to bf16
        # before the final linear layer
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits, state


if __name__ == "__main__":
    from subnet.shared_bt.model.configs import LLAMA32_CONFIG_1B

    cfg = LLAMA32_CONFIG_1B

    bottleneck_positions = [8]

    model = Llama3ModelNativeBottlenecks(cfg, bottleneck_positions=bottleneck_positions, device="cuda")

    print(model)

    # Create fake batch of longs
    in_idx = torch.randint(0, cfg["vocab_size"], (10, 800)).to("cuda")
    logits, state = model(in_idx)

    print(logits.shape)
    print(state)

    # Create fake batch of short input
