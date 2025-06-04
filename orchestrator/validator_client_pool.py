import asyncio
import settings
import aiohttp
import random

from loguru import logger
from pydantic import BaseModel, Field, model_validator
from aiohttp import ClientError, ContentTypeError
from typing import List, Dict, Any, Optional, Literal, Tuple
import bittensor as bt
from utils.epistula import generate_header, create_message_body


class ValidatorClient(BaseModel):
    """
    A client for a validator.

    Args:
        host (str): The host of the validator
        port (int): The port of the validator
        scheme (str): The scheme of the validator
        uid (int): The UID of the validator
        hotkey (str): The hotkey of the validator
    """

    session: aiohttp.ClientSession | None = None
    max_retries: int = 3
    retry_delay: float = 1.0
    host: str
    port: int
    scheme: str = "http"
    available: bool = True
    tracked_miner_hotkey: str | None = None
    hotkey: str | None = None
    base_url: str | None = None
    wallet: bt.wallet | None = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def set_base_url(self):
        self.base_url = f"{self.scheme}://{self.host}:{self.port}"
        return self

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        if not settings.VALIDATE:
            return {"is_valid": True, "score": 0.0, "reason": "Validation disabled"}

        if not self.wallet or not self.wallet.hotkey:
            raise ValueError("Wallet and hotkey must be set for API requests")

        # Create message body for signing
        body = kwargs.get("json", {})
        body_bytes = create_message_body(body)
        # Generate Epistula headers using hotkey
        headers = generate_header(self.wallet.hotkey, body_bytes)
        # Add headers to request
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].update(headers)

        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                async with getattr(self.session, method)(url, **kwargs) as response:
                    if response.status >= 500:
                        logger.warning(
                            f"Server error {response.status} on attempt {attempt + 1} for url {url} and params {kwargs}"
                        )
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue

                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        # Mark this validator as unavailable
                        self.available = False
                        raise aiohttp.ClientError(f"Request failed with status {response.status}: {error_text}")

                    try:
                        return await response.json()
                    except ContentTypeError:
                        error_text = await response.text()
                        logger.error(f"Failed to parse JSON response: {error_text}")
                        raise aiohttp.ClientError(f"Failed to parse JSON response: {error_text}")

            except (ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"Request for method {method} url {url} and params {kwargs} failed on attempt {attempt + 1}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

                # Mark this validator as unavailable after all retries
                self.available = False
                raise

        self.available = False
        raise aiohttp.ClientError(f"Failed after {self.max_retries} attempts")

    async def initialize_validator(
        self, miner_hotkey: str, layer: int, weight_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initialize the gradient validator to track a specific miner and layer."""
        data = {"miner_hotkey": miner_hotkey, "layer": layer, "weight_path": weight_path}

        return await self._make_request("post", "/gradient-validator/initialize", json=data)

    async def forward_activation(
        self, activation_uid: str, direction: Literal["forward", "backward", "initial"]
    ) -> Dict[str, Any]:
        """Perform a forward pass with the gradient validator."""
        params = {"activation_uid": activation_uid, "direction": direction}
        return await self._make_request("post", "/gradient-validator/forward", params=params)

    async def backward_activation(self, activation_uid: str) -> Dict[str, Any]:
        """Perform a backward pass with the gradient validator."""
        params = {"activation_uid": activation_uid}
        return await self._make_request("post", "/gradient-validator/backward", params=params)

    async def validate_weights(
        self, weights_path: str, metadata_path: str, optimizer_state_path: str
    ) -> Dict[str, Any]:
        """Validate the weights submitted by a miner."""
        logger.debug(f"Validating weights for miner {self.tracked_miner_hotkey}")
        params = {
            "weights_path": weights_path,
            "metadata_path": metadata_path,
            "optimizer_state_path": optimizer_state_path,
        }
        return await self._make_request("post", "/gradient-validator/validate-weights", params=params)

    async def get_validator_status(self) -> Dict[str, Any]:
        """Get the current status of the gradient validator."""
        return await self._make_request("get", "/gradient-validator/status")


class ValidatorClientPool(BaseModel):
    validators: List[ValidatorClient] = Field(default_factory=list)
    initialized: bool = False
    wallet: bt.wallet | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def validator_count(self):
        return len(self.validators)

    async def get_tracked_miners(self) -> List[str]:
        """Get a list of all tracked miners."""
        return [v.tracked_miner_hotkey for v in self.validators if v.tracked_miner_hotkey is not None]

    async def close(self):
        """Close all validator client connections."""
        for validator in self.validators:
            await validator.__aexit__(None, None, None)

    def __len__(self):
        return len(self.validators)

    def get_available_validators(self) -> List[ValidatorClient]:
        """Return a list of available validators."""
        return [v for v in self.validators if v.available and v.tracked_miner_hotkey is None]

    def get_random_validator(self) -> ValidatorClient | None:
        """Get a random available validator."""
        available = self.get_available_validators()
        if not available:
            logger.warning("No validators available")
            return None

        return random.choice(available)

    async def assign_miner_to_validator(self, miner_hotkey: str, layer: int, weight_path: str) -> Tuple[bool, str]:
        """Assign a miner to be tracked by a validator.

        Args:
            miner_hotkey (str): The hotkey of the miner to assign
            layer (int): The layer of the miner to assign
            weight_path (str): The path to the weights file

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success and the hotkey of the validator
        """

        validator = self.get_random_validator()
        if not validator:
            return False, "no_validator_available"

        try:
            await validator.initialize_validator(miner_hotkey=miner_hotkey, layer=layer, weight_path=weight_path)
            validator.tracked_miner_hotkey = miner_hotkey
            validator.available = False
            if validator.hotkey is None:
                return True, "unknown_validator"
            return True, validator.hotkey

        except Exception as e:
            logger.error(f"Failed to assign miner {miner_hotkey} to validator: {e}")
            return False, "validator_error"

    async def validate_activation(
        self,
        activation_uid: str,
        direction: Literal["forward", "backward", "initial"],
        miner_hotkey: str,
    ) -> Dict[str, Any]:
        """Validate a forward pass.

        Args:
            activation_uid (str): The UID of the activation to validate
            direction (Literal["forward", "backward", "initial"]): The direction of the activation
            miner_hotkey (str): The hotkey of the miner to validate the activation

        Returns:
            Dict[str, Any]: A dictionary containing the validation result
        """

        if not settings.VALIDATE:
            return {"is_valid": True, "score": 0.0, "reason": "Validation disabled"}

        for validator in self.validators:
            if validator.tracked_miner_hotkey == miner_hotkey:
                try:
                    return await validator.forward_activation(activation_uid=activation_uid, direction=direction)
                except Exception as e:
                    logger.error(f"Error validating forward pass: {e}")
                    continue
            else:
                logger.error(f"Miner {miner_hotkey} not tracked by validator {validator.hotkey}")

        return {"is_valid": True, "score": 0.0, "reason": "Not tracking miner"}

    async def validate_weights(
        self, weights_path: str | None, metadata_path: str | None, optimizer_state_path: str | None, miner_hotkey: str
    ) -> Dict[str, Any]:
        """Validate the weights submitted by a miner.

        Args:
            weights_path (str): The path to the weights file
            miner_hotkey (str): The hotkey of the miner to validate the weights

        Returns:
            Dict[str, Any]: A dictionary containing the validation result
        """
        try:
            logger.debug(f"Validating weights for miner {miner_hotkey}")
            if not settings.VALIDATE:
                logger.warning("Validation disabled, skipping weight validation")
                return {"is_valid": True, "score": 0.0, "reason": "Validation disabled"}

            if not weights_path:
                logger.error("No weights path provided, skipping weight validation")
                return {
                    "is_valid": False,
                    "score": 0.0,
                    "reason": "No weights path provided",
                }

            for validator in self.validators:
                if validator.tracked_miner_hotkey == miner_hotkey:
                    try:
                        is_valid, score, reason = await validator.validate_weights(
                            weights_path=weights_path,
                            metadata_path=metadata_path,
                            optimizer_state_path=optimizer_state_path,
                        )
                        return {"is_valid": is_valid, "score": score, "reason": reason}
                    except Exception as e:
                        logger.error(f"Error validating weights: {e}")
                        return {"is_valid": False, "score": 0.0, "reason": "Error validating weights"}

            raise ValueError(
                f"Miner {miner_hotkey} not tracked, but weight verification requested. Tracked miners: {await self.get_tracked_miners()}"
            )
        except Exception as e:
            logger.error(f"Error validating weights: {e}")

    async def add_validator(self, host: str, port: int, scheme: str = "http", hotkey: str | None = None) -> bool:
        """Add a validator to the pool.

        Args:
            host: The validator's host address
            port: The validator's port number
            scheme: The connection scheme (http/https)
            hotkey: Optional validator hotkey

        Returns:
            bool: True if validator was added successfully, False if pool is full
        """

        validator = ValidatorClient(host=host, port=port, scheme=scheme, hotkey=hotkey, wallet=self.wallet)
        if validator.hotkey in [v.hotkey for v in self.validators]:
            previous_validator = [v for v in self.validators if v.hotkey == validator.hotkey][0]
            self.validators.remove(previous_validator)

        await validator.__aenter__()
        self.validators.append(validator)

        logger.info(f"Added validator {hotkey[:8] if hotkey else 'unknown'} at {scheme}://{host}:{port} to pool")
        return True

    async def reset_validators(self):
        """Reset all validators."""
        for validator in self.validators:
            validator.available = True
            validator.tracked_miner_hotkey = None
