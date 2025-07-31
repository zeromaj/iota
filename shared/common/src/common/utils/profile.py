from __future__ import annotations

import asyncio
import time
import uuid
from pyinstrument import Profiler


class ProfilerManager:
    """Singleton helper that manages a global pyinstrument profiler instance.

    Usage:
        await ProfilerManager.instance().start()
        await ProfilerManager.instance().stop()  # -> S3 presigned URL
    """

    _instance: "ProfilerManager | None" = None

    def __init__(self) -> None:
        # Only allow instantiation through ``instance``
        self._profiler: Profiler | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    def instance(cls) -> "ProfilerManager":
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def start(self) -> bool:
        """Start the profiler if it is not already running."""
        async with self._lock:
            if self._profiler and self._profiler.is_running:
                return False

            self._profiler = Profiler(async_mode="enabled")
            self._profiler.start()
            return True

    async def status(self) -> dict:
        """Get the current status of the profiler.

        Returns
        -------
        dict
            Status information including whether profiler is running.
        """
        async with self._lock:
            is_running = self._profiler is not None and self._profiler.is_running
            return {
                "running": is_running,
                "profiler_exists": self._profiler is not None,
            }

    async def stop(self) -> str:
        """Stop the profiler and upload HTML report to S3.

        Returns
        -------
        str
            Presigned S3 URL for downloading the HTML report.
        """
        async with self._lock:
            if not self._profiler or not self._profiler.is_running:
                raise RuntimeError("Profiler not running")

            # Stop the profiler
            self._profiler.stop()

            # Get HTML content as string directly from pyinstrument
            html_content = self._profiler.output_html()

            # Generate unique object name for S3
            timestamp = int(time.time())
            object_name = f"profiler_reports/orchestrator_profile_{timestamp}_{uuid.uuid4().hex[:8]}.html"

            # Import here to avoid circular imports
            from orchestrator.storage.s3 import upload_html_content

            # Upload to S3 and get presigned URL
            presigned_url = await upload_html_content(html_content, object_name)

            # Reset internal state so a new profiler can be started later
            self._profiler = None
            return presigned_url
