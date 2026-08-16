import pytest
from unittest.mock import patch, MagicMock

from app.stt.models import SpeechToTextRequest
from app.stt.sarvam import SarvamSTTProvider
from app.stt.service import STTService

def test_sarvam_provider_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        provider = SarvamSTTProvider()
        req = SpeechToTextRequest(audio_data=b"dummy")
        resp = provider.transcribe(req)
        assert not resp.success
        assert "not configured" in resp.error

@patch("app.stt.sarvam.requests.post")
def test_sarvam_provider_success(mock_post):
    with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
        provider = SarvamSTTProvider()
        
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"transcript": "Hello world", "language_code": "en"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        
        req = SpeechToTextRequest(audio_data=b"dummy")
        resp = provider.transcribe(req)
        
        assert resp.success
        assert resp.transcript == "Hello world"
        assert resp.detected_language == "en"

@patch("app.stt.sarvam.requests.post")
def test_sarvam_provider_failure(mock_post):
    with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
        provider = SarvamSTTProvider()
        
        mock_post.side_effect = Exception("API timeout")
        
        req = SpeechToTextRequest(audio_data=b"dummy")
        resp = provider.transcribe(req)
        
        assert not resp.success
        assert "API timeout" in resp.error

def test_stt_service():
    with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
        service = STTService()
        assert isinstance(service.provider, SarvamSTTProvider)
