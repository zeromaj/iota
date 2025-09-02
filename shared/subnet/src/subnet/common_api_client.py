import asyncio

from aiohttp import ClientSession, ClientTimeout
from common import settings as common_settings
from common.settings import ORCHESTRATOR_HOST, ORCHESTRATOR_PORT, ORCHESTRATOR_SCHEMA
from common.utils.epistula import create_message_body, generate_header
from common.utils.exceptions import APIException
from common.utils.partitions import MinerPartition
from loguru import logger
from substrateinterface.keypair import Keypair

HEADER_REQUEST_ID = "X-Request-Id"


class CommonAPIClient:
    """
    Common API client for all entities.
    """

    @classmethod
    async def orchestrator_request(
        cls, method: str, path: str, body: dict | None = None, hotkey: Keypair | None = None
    ) -> dict:
        logger.opt(colors=True).debug(
            f"\n<magenta>Making orchestrator request | method: {method} | path: {path}</magenta>"
        )

        headers = None
        error = None
        request_id = None  # Will be extracted from response
        body_bytes = create_message_body(data={} if not body else body)

        if hotkey:
            headers = generate_header(hotkey, body_bytes)
            # Don't add request ID to headers - let orchestrator generate it

        for i in range(common_settings.REQUEST_RETRY_COUNT):
            try:
                if i:
                    logger.warning(f"Retrying request to endpoint {path} (attempt {i + 1})")

                timeout = ClientTimeout(total=common_settings.CLIENT_REQUEST_TIMEOUT)
                async with ClientSession(timeout=timeout) as session:
                    async with session.request(
                        method,
                        f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}{path}",
                        json=body,
                        headers=headers,
                    ) as response:
                        # Extract request ID from response headers
                        request_id = response.headers.get(HEADER_REQUEST_ID, "unknown")
                        response_text = None

                        # Add request ID to logger context for all subsequent logs
                        with logger.contextualize(request_id=request_id):
                            if response.status == 429:
                                response_text = await response.text() if not response_text else response_text
                                logger.warning(
                                    f"Rate limited on request to endpoint {path}: {response.status} - {response_text}"
                                )
                            if response.status == 404:
                                response_text = await response.text() if not response_text else response_text
                                logger.error(
                                    f"Bad request on request to endpoint {path}: {response.status} - {response_text}"
                                )
                                raise APIException(
                                    f"Bad request on request to endpoint {path}: {response.status} - {response_text}"
                                )
                            if response.status != 200:
                                # Handle non-JSON error responses
                                response_text = await response.text() if not response_text else response_text
                                if response.status == 429:
                                    logger.warning(
                                        f"Rate limited on request to endpoint {path}: {response.status} - {response_text}"
                                    )
                                else:
                                    logger.error(
                                        f"Error making orchestrator request to endpoint {path}: {response.status} - {response_text}"
                                    )
                                await asyncio.sleep(2)
                            else:
                                response_json = await response.json()
                                logger.debug(
                                    f"Successfully completed request to {path}; response: {str(response_json)[:100]}"
                                )
                                return response_json
            except Exception as e:
                # Log with request ID if we have one
                if request_id:
                    with logger.contextualize(request_id=request_id):
                        logger.error(f"Error making orchestrator request: {e}")
                else:
                    logger.error(f"Error making orchestrator request: {e}")
                error = e

        # Final error logging with request ID context if available
        if error:
            if request_id:
                with logger.contextualize(request_id=request_id):
                    logger.error(
                        f"Failed to complete request to {path} after {common_settings.REQUEST_RETRY_COUNT} attempts"
                    )
            raise error
        else:
            error_msg = f"Got bad/no response from orchestrator: {response.status}, {response_text}"
            if request_id:
                with logger.contextualize(request_id=request_id):
                    logger.error(error_msg)
            raise Exception(error_msg)

    @classmethod
    async def check_orchestrator_health(cls, hotkey: Keypair) -> bool | dict:
        try:
            response = await cls.orchestrator_request(method="GET", path="/healthcheck", hotkey=hotkey)
            if "error_name" in response:
                return response
            return response
        except Exception as e:
            logger.exception(f"Error checking orchestrator health: {e}")
            raise e

    @classmethod
    async def get_merged_partitions(cls, hotkey: Keypair) -> list[MinerPartition] | dict:
        try:
            response = await cls.orchestrator_request(method="GET", path="/common/get_merged_partitions", hotkey=hotkey)
            if "error_name" in response:
                return response
            return [MinerPartition(**partition) for partition in response]
        except Exception as e:
            logger.error(f"Error getting merged partitions: {e}")
            raise
