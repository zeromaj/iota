import math
from loguru import logger
import torch

from common.utils.exceptions import NanInfException
from subnet.model.utils import _clean_gpu_memory
from common import settings as common_settings
from subnet.model.loaders import load_model_split
from subnet.utils.vector_utils import (
    add_artificial_gradients,
    check_for_nans_and_infs,
)
from subnet.model.tokenizer import load_tokenizer


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

    async def initialize_model_manager(
        self,
        model_config: dict,
        model_metadata: dict,
        model_weights: torch.Tensor,
        optimizer_state: dict,
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

        try:
            # Ensure previous model artifacts are cleared before loading a new one
            _clean_gpu_memory()

            # Check GPU memory before loading model
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(
                    f"GPU memory before model load for miner {self.logger_attributes['hotkey'][:8]}: {allocated_memory:.2f}GB / {total_memory:.2f}GB"
                )

                if allocated_memory > total_memory * 0.8:  # If more than 80% already used
                    logger.warning(f"High GPU memory usage detected before model load: {allocated_memory:.2f}GB")

            # Load a newly initialized model (ie: has random weights)
            await self._load_model(layer=layer)
            await self._load_optimizer()

            # Load the model weights and optimizer state
            logger.info(
                f"⏳ Setting model weights and optimizer state for layer {self.layer} for miner {self.logger_attributes['hotkey'][:8]} on initialization"
            )
            await self.set_model_weights_and_optimizer_state(
                model_weights=model_weights, optimizer_state=optimizer_state
            )

            # Load the tokenizer and vocab info if this is the first or last layer
            if layer == 0 or layer == self.model_metadata["n_splits"] - 1:
                self.tokenizer = load_tokenizer(tokenizer_name=self.model_metadata["tokenizer_name"])
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

    async def clip_gradients(self):
        total_model_params: int = sum(p.numel() for p in self.model.parameters())
        logger.debug(f"Total model params: {total_model_params}")

        split_grad_norm = self.model_metadata["grad_clip_norm"] * math.sqrt(
            total_model_params / self.model_config["total_global_params"]
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
        if model_weights is not None and optimizer_state is not None:
            torch.nn.utils.vector_to_parameters(model_weights, self.model.parameters())  # inplace operation.
            self.optimizer.load_state_dict(optimizer_state)
        elif model_weights is None and optimizer_state is not None:
            raise Exception("Model weights must be provided if optimizer state is provided")
        elif model_weights is not None and optimizer_state is None:
            raise Exception("Optimizer state must be provided if model weights are provided")
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
