import os
import uvicorn
import settings
from loguru import logger

from fastapi import FastAPI
from orchestrator.api import router as orchestrator_router
from storage.api import router as storage_router
import sys
from prometheus_fastapi_instrumentator import Instrumentator

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="DEBUG")  # Add stderr handler for terminal output
if os.path.exists("orchestrator.log"):
    try:
        os.remove("orchestrator.log")
        logger.info("Removed existing orchestrator.log file")
    except OSError as e:
        logger.error(f"Error removing miners.log: {e}")

logger.add(
    "orchestrator.log",
    rotation="5 MB",  # Rotate at 5MB
    level="DEBUG",
    retention=1,  # Keep only latest log file
)


app = FastAPI()

# Add Prometheus instrumentation if enabled
if os.getenv("PROMETHEUS", "") != "":
    Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(orchestrator_router)
app.include_router(storage_router)


if __name__ == "__main__":
    # for some reason model merge takes really long to launch, but we need to make sure it goes online before the orchestrator starts
    try:
        # Run the main orchestrator
        uvicorn.run(app, host=settings.ORCHESTRATOR_HOST, port=settings.ORCHESTRATOR_PORT)
    except Exception as e:
        logger.exception(f"Error starting orchestrator: {e}")
        raise e
