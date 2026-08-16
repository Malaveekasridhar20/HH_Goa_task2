from abc import ABC, abstractmethod
from typing import Optional
from .models import SpeechToTextRequest, SpeechToTextResponse

class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, request: SpeechToTextRequest) -> SpeechToTextResponse:
        """Transcribe audio data to text."""
        pass
