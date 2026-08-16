import os
import time
import requests
import logging
from typing import Optional

from .models import SpeechToTextRequest, SpeechToTextResponse
from .provider import STTProvider

logger = logging.getLogger(__name__)

class SarvamSTTProvider(STTProvider):
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.model = os.getenv("SARVAM_MODEL", "saaras:v3")
        self.language = os.getenv("SARVAM_LANGUAGE", "hi-IN")
        self.endpoint = os.getenv("SARVAM_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
        
    def _get_language_code(self, lang: str) -> str:
        lang_map = {
            "en": "en-IN",
            "hi": "hi-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "ml": "ml-IN"
        }
        return lang_map.get(lang, "en-IN")

        
    def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        t0 = time.time()
        
        if not self.api_key:
            return SpeechToTextResponse(
                transcript="",
                provider="sarvam",
                latency=time.time() - t0,
                success=False,
                error="SARVAM_API_KEY not configured"
            )
            
        try:
            headers = {
                "api-subscription-key": self.api_key
            }
            
            # The API requires a file object with a filename.
            # We will use 'audio.wav' since our audio is in WAV format.
            files = {
                "file": ("audio.wav", request.audio_data, "audio/wav")
            }
            
            data = {
                "model": self.model
            }
            
            if request.language_hint:
                data["language_code"] = self._get_language_code(request.language_hint)
            
            # Use request's language_hint or default

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    resp = requests.post(self.endpoint, headers=headers, files=files, data=data, timeout=15)
                    resp.raise_for_status()
                    break
                except requests.exceptions.HTTPError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Sarvam HTTP Error: {e.response.text}")
                        raise e
                    time.sleep(0.5)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.5)
            
            result_data = resp.json()
            # Depending on Sarvam's exact response structure:
            # Usually: {"transcript": "..."}
            transcript = result_data.get("transcript", "")
            
            return SpeechToTextResponse(
                transcript=transcript,
                detected_language=result_data.get("language_code"),
                provider="sarvam",
                latency=time.time() - t0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Sarvam STT failed: {str(e)}")
            return SpeechToTextResponse(
                transcript="",
                provider="sarvam",
                latency=time.time() - t0,
                success=False,
                error=str(e)
            )
