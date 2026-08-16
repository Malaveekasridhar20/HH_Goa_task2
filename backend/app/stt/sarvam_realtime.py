import os
import time
import asyncio
import json
import base64
import websockets
import logging
from .models import SpeechToTextRequest, SpeechToTextResponse
from .provider import STTProvider

logger = logging.getLogger(__name__)

class SarvamRealtimeSTTProvider(STTProvider):
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.model = "saaras:v3-realtime"
        
    def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        # Fulfills interface but real usage for this experiment is async.
        return asyncio.run(self.transcribe_async(request))
        
    async def transcribe_async(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        # A single-shot cold connection using realtime manual endpointing
        t0 = time.time()
        if not self.api_key:
            return SpeechToTextResponse(transcript="", provider="sarvam_realtime", latency=0, success=False, error="No key")
            
        url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?model={self.model}&endpointing=manual&language_code={request.language_hint or 'hi-IN'}"
        headers = {"api-subscription-key": self.api_key}
        
        final_text = ""
        
        try:
            async with websockets.connect(url, additional_headers=headers) as websocket:
                # Signal speech start
                await websocket.send(json.dumps({"event": "speech_start"}))
                
                # We expect actual wave chunking in evaluate script, but here we just send chunks fast
                chunk_size = 4096
                chunks = [request.audio_data[i:i+chunk_size] for i in range(0, len(request.audio_data), chunk_size)]
                for chunk in chunks:
                    await websocket.send(json.dumps({
                        "event": "audio_input",
                        "audio": base64.b64encode(chunk).decode('utf-8')
                    }))
                    await asyncio.sleep(0.01)
                    
                await websocket.send(json.dumps({"event": "flush"}))
                await websocket.send(json.dumps({"event": "speech_end"}))
                
                # Listen for the final transcript
                while True:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(msg)
                        if data.get("event") == "transcript":
                            t = data.get("text", "").strip()
                            # It's manual endpointing. "is_final" might be True after flush/speech_end
                            if data.get("is_final", False) or data.get("final", False):
                                final_text = t
                                break
                            else:
                                final_text = t # fallback if loop breaks unexpectedly
                        elif data.get("event") == "error":
                            break
                    except asyncio.TimeoutError:
                        break
                        
        except Exception as e:
            logger.error(f"Realtime error: {e}")
            return SpeechToTextResponse(transcript="", provider="sarvam_realtime", latency=time.time()-t0, success=False, error=str(e))
            
        return SpeechToTextResponse(transcript=final_text, provider="sarvam_realtime", latency=time.time()-t0, success=True)
