import gc
import math

from common.utils.exceptions import NanInfException
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
        self.total_model_params = None
        self.optimizer_step_count: int = 0
        self.backwards_since_reset: int = 0

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

            # Load the tokenizer and vocab info if this is the first or last layer
            if layer == 0 or layer == common_settings.N_LAYERS - 1:
                await self._load_tokenizer()
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
                    output_activations,
                    f"output activations for miner {self.logger_attributes['hotkey'][:8]}",
                    exception_type=NanInfException,
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
        if self.total_model_params is None:
            self.total_model_params = sum(p.numel() for p in self.model.parameters())
            logger.debug(f"Total model params: {self.total_model_params}")

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
                        1,
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
            self.model.parameters(),
            lr=common_settings.LEARNING_RATE,
            weight_decay=common_settings.WEIGHT_DECAY,
            betas=(common_settings.BETAS[0], common_settings.BETAS[1]),
            eps=common_settings.EPS,
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

        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info(f"loaded vocab info: vocab size | {self.vocab_size} | EOS token id | {self.eos_token_id}")

    async def _load_tokenizer(self):
        logger.info(f"Loading tokenizer from {common_settings.TOKENIZER_NAME}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(common_settings.TOKENIZER_NAME, token=common_settings.HF_TOKEN)
            if tokenizer is None:
                raise Exception("Error loading tokenizer")

            self.tokenizer = tokenizer

        except Exception as e:
            logger.exception(f"Error loading tokenizer: {e}")
            raise

    async def local_all_reduce(self, learning_rate: float):
        """local all reduce is for local testing purposes. It's not used in the production code.

        Args:
            layer (int): the layer to all reduce
        """
        logger.info(f"{self.logger_attributes['hotkey'][:8]} is beginning local all reduce {self.optimizer_step_count}")
        self.optimizer_step_count += 1

        # Check gradients for nans and infs
        # Flatten gradients into a 1D tensor
        flat_params = torch.nn.utils.parameters_to_vector(self.model.parameters())

        check_for_nans_and_infs(
            flat_params,
            f"model parameters of len({len(flat_params)}) for miner {self.logger_attributes['hotkey'][:8]} before clipping",
            exception_type=NanInfException,
        )
        # Clip the gradients
        await self.clip_gradients()

        # flat_gradients = torch.nn.utils.parameters_to_vector([p.grad for p in self.model.parameters()])
        flat_params = torch.nn.utils.parameters_to_vector(self.model.parameters())

        # request the learning rate from the orchestrator
        self.optimizer.param_groups[0]["lr"] = learning_rate
        logger.debug(f"Stepping optimizer for miner {self.logger_attributes['hotkey'][:8]}")
        self.optimizer.step()

        # self.lr_scheduler.step()
        logger.info(f"{self.logger_attributes['hotkey'][:8]} learning rate: {self.optimizer.param_groups[0]['lr']}")

        # Zero the gradients
        self.optimizer.zero_grad()

    def reset(self):
        """Needs to reset all the attributes of the class"""

        # Need to delete these because of memory concerns.
        del self.model
        self.model = None
        del self.optimizer
        self.optimizer = None
        del self.tokenizer
        self.tokenizer = None

        self.vocab_size = None
        self.eos_token_id = None
        self.layer = None
        self.device = None
        self.logger_attributes = None

        # clear all the gpu memory and all torch related objects
        self._clean_gpu_memory()
