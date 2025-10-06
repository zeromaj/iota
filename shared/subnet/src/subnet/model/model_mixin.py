import math

from subnet.utils.partition_utils import load_model_weights
import torch
from common import settings as common_settings
from common.utils.exceptions import NanInfException
from loguru import logger
from subnet.model.loaders import load_model_split
from subnet.model.tokenizer import load_tokenizer
from subnet.model.utils import _clean_gpu_memory, log_cuda_memory_usage
from subnet.utils.vector_utils import add_artificial_gradients, check_for_nans_and_infs


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
        return super().parameters()


class ModelManager:
    def __init__(self):
        self.model_config: dict | None = None
        self.model_metadata: dict | None = None
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.vocab_size: int | None = None
        self.eos_token_id: int | None = None
        self.layer: int | None = None
        self.device: str | None = None
        self.logger_attributes: dict | None = None
        self.optimizer_step_count: int = 0
        self.epoch_on_registration: int = 0
        self.epoch_counter: int = 0

    async def initialize_model_manager(
        self,
        model_config: dict,
        model_metadata: dict,
        model_weights: torch.Tensor | None,
        optimizer_state: dict | None,
        layer: int,
        device: str,
        logger_attributes: dict,
    ):
        """Initializes the model, weights, optimizer, tokenizer, and vocab info
        for the layer specified.

        Args:
            model_config (dict): The model config to set.
            model_metadata (dict): The model metadata to set.
            model_weights (torch.Tensor): The model weights to set. If None, the model will be initialized with random weights.
            optimizer_state (dict): The optimizer state to set. If None, the optimizer will be initialized with random state.
            layer (int): The layer to initialize
            device (str): The device to initialize the model on
            logger_attributes (dict): The logger attributes to set
        """
        self.model_config = model_config
        self.model_metadata = model_metadata
        self.layer = layer
        self.device = device
        self.logger_attributes = logger_attributes

        assert isinstance(self.model_config, dict), "Model config must be a dict"
        assert isinstance(self.model_metadata, dict), "Model metadata must be a dict"

        with logger.contextualize(gpu="initialize model manager"):
            try:
                # Ensure previous model artifacts are cleared before loading a new one
                _clean_gpu_memory()

                # Check GPU memory before loading model
                log_cuda_memory_usage(note="before model load")

                # Load a newly initialized model (ie: has random weights)
                await self._load_model(layer=layer)
                await self._load_optimizer()

                # Load the model weights and optimizer state
                logger.info(
                    f"⏳ Setting model weights and optimizer state for layer {self.layer} for miner {self.logger_attributes['hotkey'][:8]} on initialization"
                )
                if optimizer_state is not None:
                    await self.set_model_weights_and_optimizer_state(
                        model_weights=model_weights, optimizer_state=optimizer_state
                    )
                else:
                    logger.warning(
                        f"No optimizer state provided for miner on initialization: {self.logger_attributes['hotkey'][:8]}"
                    )

                # Load the tokenizer and vocab info if this is the first or last layer
                if layer == 0 or layer == self.model_metadata["n_splits"] - 1:
                    self.tokenizer = load_tokenizer(tokenizer_name=self.model_metadata["tokenizer_name"])
                    await self._load_vocab_info()

                # Final memory check after loading
                log_cuda_memory_usage(note="after model load")

                logger.success(f"✅ Model loaded successfully {self.logger_attributes['hotkey'][:8]}")

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise

    async def _forward(self, layer: int, input_activations: torch.Tensor) -> tuple[torch.Tensor, dict]:
        with logger.contextualize(gpu="forward pass"):
            log_cuda_memory_usage(note="before forward pass")

            if layer > 0:
                input_activations.requires_grad_(True)

            output_activations, state = self.model(input_activations)

            logger.info(
                f"output activations with shape {output_activations.shape} for {self.logger_attributes['hotkey'][:8]} on layer {layer}"
            )

            log_cuda_memory_usage(note="after forward pass")
            return output_activations, state

    async def _backward(
        self,
        layer: int,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        with logger.contextualize(gpu="backward pass"):
            log_cuda_memory_usage(note="before backward pass")

            # If this is the last layer, then output_activations is the loss
            if layer == self.model_metadata["n_splits"] - 1:
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

            log_cuda_memory_usage(note="after backward pass")

    async def clip_gradients(self):
        total_model_params: int = sum(p.numel() for p in self.model.parameters())
        logger.debug(f"Total model params: {total_model_params}")

        split_grad_norm = self.model_metadata["grad_clip_norm"] * math.sqrt(
            total_model_params / self.model_config["total_global_params"]
        )

        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=split_grad_norm)

    async def clip_pseudo_gradients(self, pseudo_gradients: torch.Tensor, eps: float = 1e-6):
        """
        Clips a flat pseudo gradient tensor
        """
        # Compute L2 norm of the pseudo gradient tensor
        current_grad_norm = pseudo_gradients.norm(2).item()

        total_model_params: int = sum(p.numel() for p in self.model.parameters())

        max_grad_norm = self.model_metadata["grad_clip_norm"] * math.sqrt(
            total_model_params / self.model_config["total_global_params"]
        )

        if current_grad_norm > max_grad_norm:
            logger.debug(
                f"Clipping pseudo gradients: Current grad norm: {current_grad_norm}, max grad norm: {max_grad_norm}"
            )

            # Scale down proportionally
            scale = max_grad_norm / (current_grad_norm + eps)
            pseudo_gradients = pseudo_gradients * scale
        else:
            logger.debug(
                f"No need to clip pseudo gradients: Current grad norm: {current_grad_norm}, max grad norm: {max_grad_norm}"
            )

        return pseudo_gradients

    async def _load_model(self, layer: int):
        """
        Loads the model for the layer specified.
        """
        if common_settings.MOCK:
            logger.info("Mock mode enabled - loading mock model")
            self.model = MockModel()
            self.model.train()
            return

        logger.info(f"MODEL_SPLITS: {self.model_metadata['model_splits']}")
        logger.info(f"Loading model from {self.model_config['model_name']}")

        if isinstance(self.model_config["dtype"], str):
            self.model_config["dtype"] = getattr(torch, self.model_config["dtype"].split(".")[-1])

        try:
            self.model = load_model_split(
                model_cfg=self.model_config,
                model_split=self.model_metadata["model_splits"][layer],
                device=self.device,
                seed=42,
            )
            # put the model in train mode
            self.model.train()

            # forward pass to populate bottleneck decoder in the case where
            # the bottleneck dynamically changes it size based on the input data.
            if layer > 0:
                logger.success(f"Populating bottleneck decoder for layer {layer}")
                blank_tensor = torch.zeros(
                    1,
                    common_settings.SEQUENCE_LENGTH,
                    self.model_config["bottleneck_dim"] or self.model_config["emb_dim"],
                    dtype=self.model_config["dtype"],
                ).to(self.device)

                self.model.forward(blank_tensor)

        except ValueError as e:
            logger.exception(f"{e}")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")

        # log the number of parameters
        logger.info(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters()) / 1e9}B")

    async def set_model_weights_and_optimizer_state(
        self, model_weights: torch.Tensor = None, optimizer_state: dict = None
    ):
        """
        sets the model weights and optimizer state for the layer specified.
        """

        # Ensure that both model weights and optimizer state are provided.
        if model_weights is not None:
            torch.nn.utils.vector_to_parameters(model_weights, self.model.parameters())  # inplace operation.
        else:
            logger.info("No model weights provided, keeping random weights! 🎲")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        else:
            logger.info("No model weights or optimizer state provided, keeping random weights! 🎲")

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

    async def local_optimization_step(self, learning_rate: float):
        """Perform a local optimization step every 32 backward passes."""

        with logger.contextualize(gpu="local optimization step"):
            logger.info(f"{self.logger_attributes['hotkey'][:8]} is beginning local optimization step")
            log_cuda_memory_usage(note="before local optimization step")

            # Clip the gradients
            await self.clip_gradients()

            log_cuda_memory_usage(note="after clipping gradients")

            # Step the optimizer
            if learning_rate is None:
                logger.error("Learning rate is None")
                learning_rate = common_settings.LEARNING_RATE
            logger.debug(f"Setting learning rate to {learning_rate}")
            self.optimizer.param_groups[0]["lr"] = learning_rate
            logger.debug(f"Stepping optimizer for miner {self.logger_attributes['hotkey'][:8]}")

            # Step and zero the gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

            log_cuda_memory_usage(note="after stepping optimizer")

            # TODO: Remove this once we have a better way to handle local optimization step.
            # If a miner registers at a later epoch that epoch = 1, their local optimizer can be completely bogus.
            # This is a "warm up" period, where a miner can continue to do work, but we just *dont* up date their local weights.
            if self.epoch_counter <= 2 and self.epoch_on_registration > 1:
                # load our previous weights into memory
                logger.info(
                    f"Keeping previous weights for miner {self.logger_attributes['hotkey'][:8]} with epoch counter {self.epoch_counter} and epoch on registration {self.epoch_on_registration}"
                )
                loaded_weights = load_model_weights(
                    hotkey=self.logger_attributes["hotkey"],
                    run_id=self.logger_attributes["run_id"],
                    layer_idx=self.layer,
                )
                torch.nn.utils.vector_to_parameters(loaded_weights, self.model.parameters())

            logger.info(f"{self.logger_attributes['hotkey'][:8]} completed local optimization step")
            log_cuda_memory_usage(note="after local optimization step")

    def reset(self):
        """Needs to reset all the attributes of the class"""
        with logger.contextualize(gpu="reset"):
            log_cuda_memory_usage(note="before reset")

            # Need to delete these because of memory concerns.
            del self.model
            del self.optimizer
            self.model = None
            self.optimizer = None

            self.vocab_size = None
            self.eos_token_id = None
            self.layer = None
            self.device = None
            self.logger_attributes = None

            # clear all the gpu memory and all torch related objects
            _clean_gpu_memory()
