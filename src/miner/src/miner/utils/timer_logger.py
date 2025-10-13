import json
import time
from typing import Any, Dict, Optional
from loguru import logger


class TimerLogger:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        self.enter_time: Optional[float] = None
        self.exit_time: Optional[float] = None

    async def __aenter__(self):
        self.enter_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exit_time = time.time()

        log_data = {
            "name": self.name,
            "enter_time": self.enter_time,
            "exit_time": self.exit_time,
            "metadata": self.metadata,
        }

        logger.debug(f"Timer Logger: {json.dumps(log_data)}")
        return False
