import os
from .models import SpeechToTextRequest, SpeechToTextResponse
from .provider import STTProvider
from .sarvam import SarvamSTTProvider

class STTService:
    def __init__(self, provider: STTProvider = None):
        if provider:
            self.provider = provider
        else:
            # Default to Sarvam
            self.provider = SarvamSTTProvider()
            
    def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        return self.provider.transcribe(request)
