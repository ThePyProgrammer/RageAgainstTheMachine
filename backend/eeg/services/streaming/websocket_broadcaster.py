"""WebSocket broadcaster for real-time EEG data."""

import asyncio
from shared.config.logging import get_logger

logger = get_logger(__name__)


class WebSocketBroadcaster:
    """Manages WebSocket clients and broadcasts EEG data."""

    def __init__(self):
        self.clients = []
        self.loop = None
        self._broadcast_count = 0

    def register_client(self, websocket):
        """Register a WebSocket client to receive streaming data."""
        self.clients.append(websocket)
        logger.info("[Broadcaster] Client registered. Total: %d", len(self.clients))
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
                logger.info("[Broadcaster] Event loop captured")
            except RuntimeError:
                logger.warning("[Broadcaster] No running event loop available")

    def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info("[Broadcaster] Client unregistered. Total: %d", len(self.clients))

    def broadcast(self, rows):
        """Broadcast EEG data to all connected WebSocket clients."""
        if not self.clients or self.loop is None:
            if self._broadcast_count == 0:
                logger.warning("[Broadcaster] broadcast() called but clients=%d loop=%s",
                               len(self.clients), self.loop is not None)
                self._broadcast_count = 1
            return

        payload = {"samples": rows}

        if self._broadcast_count % 250 == 0:
            logger.debug("[Broadcaster] Sending %d rows to %d clients (total broadcasts: %d)",
                         len(rows), len(self.clients), self._broadcast_count)
        self._broadcast_count += 1

        for client in self.clients[:]:
            try:
                asyncio.run_coroutine_threadsafe(client.send_json(payload), self.loop)
            except Exception as e:
                logger.error("[Broadcaster] Failed to send to client: %s", e)
                self.unregister_client(client)

    def broadcast_error(self, error_message: str):
        """Broadcast an error message to all connected WebSocket clients."""
        if not self.clients or self.loop is None:
            return

        payload = {"error": error_message}

        for client in self.clients[:]:
            try:
                asyncio.run_coroutine_threadsafe(client.send_json(payload), self.loop)
            except Exception:
                self.unregister_client(client)
