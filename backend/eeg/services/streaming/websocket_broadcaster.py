"""WebSocket broadcaster for real-time EEG data."""

import asyncio


class WebSocketBroadcaster:
    """Manages WebSocket clients and broadcasts EEG data."""

    def __init__(self):
        self.clients = []
        self.loop = None

    def register_client(self, websocket):
        """Register a WebSocket client to receive streaming data."""
        self.clients.append(websocket)
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        if websocket in self.clients:
            self.clients.remove(websocket)

    def broadcast(self, rows):
        """Broadcast EEG data to all connected WebSocket clients."""
        if not self.clients or self.loop is None:
            return

        payload = {"samples": rows}

        for client in self.clients[:]:
            try:
                asyncio.run_coroutine_threadsafe(client.send_json(payload), self.loop)
            except Exception:
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
