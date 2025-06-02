from pydantic import BaseModel, Field, IPvAnyAddress, UUID4
from typing import Literal
from datetime import datetime
from uuid import uuid4


class LossMessage(BaseModel):
    """
    Loss message sent every 1 minute.
    Contains accuracy ratings and throughput information.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    loss: float
    perplexity: float
    activation_count: int = Field(description="The current activation count (e.g. 82943th activation)")
    number_of_sampled_activations: int = Field(description="How many activations this is sampled over (throughput)")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-05-06T12:00:00",
                "loss": 7.23,
                "perplexity": 0.2,
                "activation_count": 82943,
                "number_of_sampled_activations": 1000,
            }
        }


class SyncUpdateMessage(BaseModel):
    """
    Sync/update message sent after every "effective" batch.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    activation_count: int
    number_of_sampled_activations: int = Field(description="This is the closest thing we have to batch size")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-05-06T12:00:00",
                "activation_count": 85000,
                "number_of_sampled_activations": 1500,
            }
        }


class CountdownMessage(BaseModel):
    """
    Countdown message showing real-time 'tube graph' of activations firing between miners.
    This is event-based (not persistent data) and should not be stored.
    """

    miner_hotkey: str
    activation_uid: str
    previous_miner_hotkey: str
    action: Literal["forward", "backwards"]

    class Config:
        json_schema_extra = {
            "example": {
                "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "activation_uid": "act-12345",
                "previous_miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "action": "forward",
            }
        }


class MinerStateMessage(BaseModel):
    """
    Miner State message triggered when a miner moves or registers.
    Sent once per 30 min per miner.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    layer: int = Field(description="Layer in the network (2-5 layers initially)")
    unique_miner_id: UUID4 = Field(default_factory=uuid4, description="UUID4 that changes upon deregistration")
    coldkey: str
    hotkey: str
    number_of_processed_activations: int
    ip_address: IPvAnyAddress
    is_trusted: Literal[0, 1]
    time_to_trusted: int = Field(description="How long until miner is trusted and starts getting paid")
    uptime: int = Field(description="Uptime in seconds")
    throughput: float = Field(description="Tokens per minute, typically 0-5")
    reward: float = Field(description="Reward amount (might duplicate incentive)")
    incentive: float

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-05-06T12:00:00",
                "layer": 2,
                "unique_miner_id": "123e4567-e89b-12d3-a456-426614174000",
                "coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
                "number_of_processed_activations": 15000,
                "ip_address": "192.168.1.1",
                "is_trusted": 1,
                "time_to_trusted": 3600,
                "uptime": 86400,
                "throughput": 3.5,
                "reward": 0.25,
                "incentive": 0.30,
            }
        }
