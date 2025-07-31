import gc
import math

import torch
from common import settings as common_settings
from loguru import logger
from subnet.model.loaders import load_model_split
from subnet.utils.vector_utils import (
    add_artificial_gradients,
    check_for_nans_and_infs,
)
from transformers import AutoTokenizer


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 64).to(torch.bfloat16)
        self.layer2 = torch.nn.Linear(64, 32).to(torch.bfloat16)
        self.layer3 = torch.nn.Linear(32, 100).to(torch.bfloat16)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x, {}

    def backward(
        self,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        # Pass in activation_grads to backward() to avoid implicit scalar gradient error
        output_activations.backward(activation_grads)

    def parameters(self):
        return [p for p in super().parameters()]


class ModelManager:
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.vocab_size: int | None = None
        self.eos_token_id: int | None = None
        self.layer: int | None = None
        self.device: str | None = None
        self.logger_attributes: dict | None = None

    async def initialize_model_manager(self, layer: int, device: str, logger_attributes: dict):
        """
        Initializes the model, weights, optimizer, tokenizer, and vocab info
        for the layer specified.
        """
        self.layer = layer
        self.device = device
        self.logger_attributes = logger_attributes

        try:
            # Ensure previous model artifacts are cleared before loading a new one
            self._clean_gpu_memory()

            # Check GPU memory before loading model
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(
                    f"GPU memory before model load for miner {self.logger_attributes['hotkey'][:8]}: {allocated_memory:.2f}GB / {total_memory:.2f}GB"
                )

                if allocated_memory > total_memory * 0.8:  # If more than 80% already used
                    logger.warning(f"High GPU memory usage detected before model load: {allocated_memory:.2f}GB")

            # Load the model
            await self._load_model(layer=layer)
            await self._load_optimizer()

            # Load the tokenizer if this is the first
            if layer == 0 or layer == common_settings.N_LAYERS - 1:
                await self._load_tokenizer()

            # If this is the first or last stage, get the vocab info
            if layer == 0 or layer == common_settings.N_LAYERS - 1:
                await self._load_vocab_info()

            # Final memory check after loading
            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(
                    f"GPU memory after model load for miner {self.logger_attributes['hotkey'][:8]}: {allocated_memory:.2f}GB"
                )

            logger.success(f"✅ Model loaded successfully {self.logger_attributes['hotkey'][:8]}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _clean_gpu_memory(self):
        """Force cleanup of GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all in-flight kernels
            torch.cuda.empty_cache()  # release unused cached blocks
            torch.cuda.synchronize()  # (optional) make sure the allocator work is finished

        gc.collect()
        logger.debug(f"Miner GPU memory cleaned. memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    async def _forward(self, layer: int, input_activations: torch.Tensor):
        if layer > 0:
            input_activations.requires_grad_(True)
        self.model.to(self.device)
        output_activations = self.model(input_activations.to(self.device))
        return output_activations

    async def _backward(
        self,
        layer: int,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        # If this is the last layer, then output_activations is the loss
        if layer == common_settings.N_LAYERS - 1:
            try:
                check_for_nans_and_infs(
                    output_activations, f"output activations for miner {self.logger_attributes['hotkey'][:8]}"
                )
                output_activations.backward()
            except RuntimeError as e:
                logger.error(f"Error during backward step: {e}")
                raise
        else:
            try:
                self.model.backward(output_activations, activation_grads, state)
            except RuntimeError as e:
                logger.error(f"Error during backward step: {e}")
                raise

    async def clip_gradients(self):
        if not hasattr(self, "total_model_params"):
            self.total_model_params = sum(p.numel() for p in self.model.parameters())

        split_grad_norm = common_settings.GRAD_CLIP_NORM * math.sqrt(
            self.total_model_params / common_settings.MODEL_CFG["total_global_params"]
        )
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=split_grad_norm)

    async def _load_model(self, layer: int):
        """
        Loads the model for the layer specified.
        """
        if common_settings.MOCK:
            logger.info("Mock mode enabled - loading mock model")
            self.model = MockModel()
            self.model.train()
            return

        logger.info(f"MODEL_SPLITS: {common_settings.MODEL_SPLITS}")
        logger.info(f"Loading model from {common_settings.MODEL_CFG['model_name']}")
        if isinstance(common_settings.MODEL_CFG["dtype"], str):
            common_settings.MODEL_CFG["dtype"] = getattr(torch, common_settings.MODEL_CFG["dtype"].split(".")[-1])

        try:
            self.model = load_model_split(
                model_cfg=common_settings.MODEL_CFG,
                model_split=common_settings.MODEL_SPLITS[layer],
                device=self.device,
                seed=42,
            )
            # put the model in train mode
            self.model.train()

            # forward pass to populate bottleneck decoder in the case where
            # the bottleneck dynamically changes it size based on the input data.
            if layer > 0:
                logger.success(f"Populating bottleneck decoder for layer {layer}")
                self.model.forward(
                    torch.zeros(
                        common_settings.BATCH_SIZE,
                        common_settings.SEQUENCE_LENGTH,
                        common_settings.MODEL_CFG["bottleneck_dim"] or common_settings.MODEL_CFG["emb_dim"],
                    ).to(self.device)
                )

        except ValueError as e:
            logger.exception(f"{e}")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")

        # log the number of parameters
        logger.info(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters()) / 1e9}B")

    async def _load_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=common_settings.LEARNING_RATE, weight_decay=common_settings.WEIGHT_DECAY
        )
        add_artificial_gradients(model=self.model, device=self.device)
        self.optimizer.step()
        self.optimizer.zero_grad()

        logger.info(
            f"Loaded optimizer with learning rate {common_settings.LEARNING_RATE} and weight decay {common_settings.WEIGHT_DECAY}"
        )

    async def _load_vocab_info(self):
        if common_settings.MOCK:
            logger.info("Mock mode enabled - using mock vocab info")
            self.vocab_size = 100
            self.eos_token_id = 1
            return

        if self.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(common_settings.TOKENIZER_NAME)
        else:
            tokenizer = self.tokenizer

        self.vocab_size = len(tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        logger.info(f"loaded vocab info: vocab size | {self.vocab_size} | EOS token id | {self.eos_token_id}")

    async def _load_tokenizer(self):
        logger.info(f"Loading tokenizer from {common_settings.TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(common_settings.TOKENIZER_NAME, token=common_settings.HF_TOKEN)

    async def local_all_reduce(self, learning_rate: float):
        """local all reduce is for local testing purposes. It's not used in the production code.

        Args:
            layer (int): the layer to all reduce
        """
        logger.info(f"{self.logger_attributes['hotkey'][:8]} is doing local all reduce")

        # Clip the gradients
        await self.clip_gradients()

        # request the learning rate from the orchestrator
        self.optimizer.param_groups[0]["lr"] = learning_rate
        self.optimizer.step()
        # self.lr_scheduler.step()
        logger.info(f"{self.logger_attributes['hotkey'][:8]} learning rate: {self.optimizer.param_groups[0]['lr']}")

        # self.completed_optim_steps += 1

        # self.backwards_since_reduce = 0
        # self.saved_forward_activations = {}

        # Log the loss and other training state information to wandb.
        # This is used for monitoring the loss during testing
        # if layer == common_settings.N_LAYERS - 1 and common_settings.USE_WANDB:
        #     logger.info(f"Miner {hotkey[:8]} is logging to wandb for layer {layer}")
        #     metrics = {"avg_loss": sum(self.losses_since_reduce) / len(self.losses_since_reduce)}
        #     await self._log_wandb(metrics)

        # Reset the losses since reduce
        # self.losses_since_reduce = []

        # Zero the gradients
        self.optimizer.zero_grad()

        # # Log GPU memory after weight update
        # if torch.cuda.is_available():
        #     allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        #     logger.debug(f"GPU memory after local all reduce: {allocated:.2f}GB")

    def reset_model_manager(self):
        """Needs to reset all the attributes of the class"""
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self.vocab_size = None
        self.eos_token_id = None
        self.layer = None
        self.device = None
        self.logger_attributes = None
