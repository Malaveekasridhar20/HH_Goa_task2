from pydantic import BaseModel
from typing import Optional

class SpeechToTextRequest(BaseModel):
    audio_data: bytes
    language_hint: Optional[str] = None
    request_id: Optional[str] = None

class SpeechToTextResponse(BaseModel):
    transcript: str
    detected_language: Optional[str] = None
    provider: str
    latency: float
    success: bool
    error: Optional[str] = None
