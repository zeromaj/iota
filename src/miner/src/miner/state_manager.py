from bittensor import Wallet
from pydantic import BaseModel


class StateManager(BaseModel):
    wallet: Wallet
    run_id: str = None
    layer: int = 0
    training_epoch_when_registered: int = None

    class Config:
        arbitrary_types_allowed = True
