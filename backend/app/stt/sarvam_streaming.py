import os
import json
import base64
import asyncio
import websockets
import logging
import time

from .models import SpeechToTextRequest, SpeechToTextResponse
from .provider import STTProvider

logger = logging.getLogger(__name__)

class SarvamStreamingSTTProvider(STTProvider):
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.model = os.getenv("SARVAM_MODEL", "saaras:v3")
        self.endpoint = "wss://api.sarvam.ai/speech-to-text/ws"
        
    def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        """
        Since STTProvider is synchronous in the REST pipeline, calling this directly
        will run it synchronously via asyncio.run. 
        For true async streaming investigation, the evaluate_streaming.py script 
        will interact with websockets directly or we can expose an async method here.
        We provide this sync wrapper just to fulfill the STTProvider interface if needed.
        """
        return asyncio.run(self.transcribe_async(request))
        
    async def transcribe_async(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        t0 = time.time()
        
        if not self.api_key:
            return SpeechToTextResponse(
                transcript="", provider="sarvam_streaming", latency=time.time() - t0, success=False, error="SARVAM_API_KEY not configured"
            )
            
        headers = {"api-subscription-key": self.api_key}
        final_transcript = ""
        
        try:
            async with websockets.connect(self.endpoint, extra_headers=headers) as websocket:
                
                # Chunk size for streaming (e.g. 50ms)
                chunk_size = 1600 * 2 # Roughly assuming PCM byte sizing if it was PCM, 
                # but request.audio_data here is likely wav/mp3.
                # We will just send it in chunks of 4096 bytes to simulate streaming.
                chunks = [request.audio_data[i:i+4096] for i in range(0, len(request.audio_data), 4096)]
                
                async def receiver():
                    nonlocal final_transcript
                    try:
                        async for message in websocket:
                            data = json.loads(message)
                            if "transcript" in data:
                                t = data["transcript"].strip()
                                if data.get("is_final", False) or data.get("final", False):
                                    final_transcript = t
                                else:
                                    final_transcript = t # fallback if no is_final emitted
                    except websockets.exceptions.ConnectionClosed:
                        pass
                
                listener = asyncio.create_task(receiver())
                
                config_msg = {
                    "audio": {
                        "sample_rate": "16000",
                        "encoding": "audio/wav"
                    },
                    "model": self.model,
                    "mode": "transcribe",
                    "language_code": request.language_hint or "hi-IN"
                }
                await websocket.send(json.dumps(config_msg))
                
                for chunk in chunks:
                    await websocket.send(chunk)
                    await asyncio.sleep(0.01)
                    
                await asyncio.sleep(0.5) # Wait for trailing responses
                await websocket.close()
                await listener
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return SpeechToTextResponse(
                transcript="", provider="sarvam_streaming", latency=time.time() - t0, success=False, error=str(e)
            )
            
        return SpeechToTextResponse(
            transcript=final_transcript, provider="sarvam_streaming", latency=time.time() - t0, success=True
        )
