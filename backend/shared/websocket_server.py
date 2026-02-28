# Generic WebSocket server
from fastapi import WebSocket
import json

from shared.config.logging import get_logger

logger = get_logger(__name__)


class WebSocketServer:
    def __init__(self, pipeline, recorder, state):
        self.pipeline = pipeline
        self.recorder = recorder
        self.state = state

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # Try to receive message (can be JSON or binary)
                message = await websocket.receive()

                # Handle different message types
                if "json" in message:
                    data = message["json"]
                elif "bytes" in message:
                    data = message["bytes"]
                elif "text" in message:
                    # Parse text as JSON
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        data = message["text"]
                else:
                    continue

                # Process data through pipeline
                processed = await self.pipeline.process(data)

                # Send response as JSON
                await websocket.send_json(processed)

                # Buffer if recording
                if self.state.get("is_recording"):
                    self.recorder.buffer(data)

        except Exception as e:
            logger.error("WebSocket error: %s", e, exc_info=True)
            await websocket.close()
            self.recorder.end_session()
