"""WebSocket broadcaster for real-time EEG data."""

import asyncio
from concurrent.futures import Future
from shared.config.logging import get_logger

logger = get_logger(__name__)


class WebSocketBroadcaster:
    """Manages WebSocket clients and broadcasts EEG data."""

    def __init__(self):
        self.clients = []
        self.loop = None
        self._broadcast_count = 0
        self._pending_sends = {}

    def register_client(self, websocket):
        """Register a WebSocket client to receive streaming data."""
        self.clients.append(websocket)
        self._pending_sends.pop(websocket, None)
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
            pending = self._pending_sends.pop(websocket, None)
            if pending and not pending.done():
                pending.cancel()
            logger.info("[Broadcaster] Client unregistered. Total: %d", len(self.clients))

    def _on_send_done(self, websocket, future: Future):
        current = self._pending_sends.get(websocket)
        if current is future:
            self._pending_sends.pop(websocket, None)
        try:
            future.result()
        except Exception as e:
            logger.error("[Broadcaster] Send failed; unregistering client: %s", e)
            self.unregister_client(websocket)

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
            pending = self._pending_sends.get(client)
            if pending and not pending.done():
                # Slow client: skip this frame to prevent unbounded queue growth.
                continue
            try:
                future = asyncio.run_coroutine_threadsafe(
                    client.send_json(payload), self.loop
                )
                self._pending_sends[client] = future
                future.add_done_callback(
                    lambda f, websocket=client: self._on_send_done(websocket, f)
                )
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
                future = asyncio.run_coroutine_threadsafe(
                    client.send_json(payload), self.loop
                )
                self._pending_sends[client] = future
                future.add_done_callback(
                    lambda f, websocket=client: self._on_send_done(websocket, f)
                )
            except Exception:
                self.unregister_client(client)
