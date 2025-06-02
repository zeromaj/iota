import aiohttp
import asyncio
from loguru import logger
from typing import Dict, Any, Literal
from aiohttp import ClientError, ContentTypeError
import settings


class GradientValidatorClient:
    def __init__(self):
        self.base_url = f"{settings.ORCHESTRATOR_SCHEME}://{settings.ORCHESTRATOR_HOST}:{settings.ORCHESTRATOR_PORT}"
        self.session = None
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
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
                raise

        raise aiohttp.ClientError(f"Failed after {self.max_retries} attempts")

    async def initialize_validator(self, miner_hotkey: str, layer: int, weight_path: str) -> Dict[str, Any]:
        """Initialize the gradient validator to track a specific miner and layer."""
        data = {"miner_hotkey": miner_hotkey, "layer": layer}
        if weight_path:
            data["weight_path"] = weight_path
        return await self._make_request("post", f"{self.base_url}/gradient-validator/initialize", json=data)

    async def forward_activation(
        self, activation_uid: str, direction: Literal["forward", "backward", "initial"]
    ) -> Dict[str, Any]:
        """Perform a forward pass with the gradient validator."""
        params = {"activation_uid": activation_uid, "direction": direction}
        return await self._make_request("post", f"{self.base_url}/gradient-validator/forward", params=params)

    async def backward_activation(self, activation_uid: str) -> Dict[str, Any]:
        """Perform a backward pass with the gradient validator."""
        params = {"activation_uid": activation_uid}
        return await self._make_request("post", f"{self.base_url}/gradient-validator/backward", params=params)

    async def validate_weights(self, weights_path: str) -> Dict[str, Any]:
        """Validate the weights submitted by a miner."""
        params = {"weights_path": weights_path}
        return await self._make_request(
            "post",
            f"{self.base_url}/gradient-validator/validate-weights",
            params=params,
        )

    async def reset_validator(self) -> Dict[str, Any]:
        """Reset the gradient validator to its initial state."""
        return await self._make_request("post", f"{self.base_url}/gradient-validator/reset")

    async def get_validator_status(self) -> Dict[str, Any]:
        """Get the current status of the gradient validator."""
        return await self._make_request("get", f"{self.base_url}/gradient-validator/status")
