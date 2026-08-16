import time
import logging
from typing import Optional

from .models import VoiceRAGRequest, VoiceRAGResponse
from app.stt.service import STTService
from app.stt.models import SpeechToTextRequest
from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator
from app.generation.generator import AnswerGenerator

logger = logging.getLogger(__name__)

class VoiceRAGPipeline:
    def __init__(self, 
                 stt_service: Optional[STTService] = None,
                 en_retriever: Optional[Retriever] = None,
                 hi_retriever: Optional[Retriever] = None,
                 ta_retriever: Optional[Retriever] = None,
                 te_retriever: Optional[Retriever] = None,
                 ml_retriever: Optional[Retriever] = None,
                 extractive_generator: Optional[ExtractiveAnswerGenerator] = None,
                 llm_generator: Optional[AnswerGenerator] = None):
                 
        self.stt_service = stt_service or STTService()
        
        # Load default retrievers if not provided (they will load index from standard paths if available)
        try:
            self.en_retriever = en_retriever or Retriever(index_dir="data/indexes/english")
            self.hi_retriever = hi_retriever or Retriever(index_dir="data/indexes/hindi")
            self.ta_retriever = ta_retriever or Retriever(index_dir="data/indexes/tamil")
            self.te_retriever = te_retriever or Retriever(index_dir="data/indexes/telugu")
            self.ml_retriever = ml_retriever or Retriever(index_dir="data/indexes/malayalam")
            self.extractive_generator = extractive_generator or ExtractiveAnswerGenerator(embedding_service=self.en_retriever.embedding_service)
        except Exception as e:
            logger.warning(f"Failed to load default indices: {e}")
            self.en_retriever = en_retriever
            self.hi_retriever = hi_retriever
            self.ta_retriever = ta_retriever
            self.te_retriever = te_retriever
            self.ml_retriever = ml_retriever
            self.extractive_generator = extractive_generator
            
        self.llm_generator = llm_generator or AnswerGenerator()
        
    def execute(self, request: VoiceRAGRequest) -> VoiceRAGResponse:
        t_total_start = time.time()
        
        # 1. Validate Audio
        if not request.audio_data or len(request.audio_data) == 0:
            return VoiceRAGResponse(
                success=False,
                transcript="",
                answer="",
                source_chunk_ids=[],
                error="Empty audio data provided",
                total_latency_ms=(time.time() - t_total_start) * 1000
            )

        # 2. STT
        stt_req = SpeechToTextRequest(
            audio_data=request.audio_data,
            language_hint=request.language_hint
        )
        
        t_stt_start = time.time()
        stt_resp = self.stt_service.transcribe(stt_req)
        stt_latency_ms = (time.time() - t_stt_start) * 1000
        
        if not stt_resp.success:
            return VoiceRAGResponse(
                success=False,
                transcript="",
                answer="",
                source_chunk_ids=[],
                stt_latency_ms=stt_latency_ms,
                total_latency_ms=(time.time() - t_total_start) * 1000,
                error=f"STT failed: {stt_resp.error}"
            )
            
        # 3. Validate Transcript (Guardrail: empty/off-topic)
        transcript = stt_resp.transcript.strip()
        if not transcript or len(transcript) < 2:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="I couldn't hear what you said.",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                total_latency_ms=(time.time() - t_total_start) * 1000,
                error="Empty or extremely short transcript"
            )

        # Choose language index based on hint or detected. Default to English for simplicity if unknown.
        lang = request.language_hint or stt_resp.detected_language or "en"
        lang = lang.lower()
        if "hi" in lang:
            retriever = self.hi_retriever
        elif "ta" in lang:
            retriever = self.ta_retriever
        elif "te" in lang:
            retriever = self.te_retriever
        elif "ml" in lang:
            retriever = self.ml_retriever
        else:
            retriever = self.en_retriever
            
        if not retriever:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                total_latency_ms=(time.time() - t_total_start) * 1000,
                error="Retriever not available for language"
            )

        # 4. Retrieval
        t_ret_start = time.time()
        try:
            # Phase 1: Real Score Fusion (Dense + BM25)
            # DENSE_WEIGHT = 0.7, BM25_WEIGHT = 0.3 as requested by optimal baseline
            chunks = retriever.retrieve_hybrid(transcript, top_k=request.top_k, dense_weight=0.7, bm25_weight=0.3)
        except Exception as e:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                retrieval_latency_ms=(time.time() - t_ret_start) * 1000,
                total_latency_ms=(time.time() - t_total_start) * 1000,
                error=f"Retrieval failed: {str(e)}"
            )
        retrieval_latency_ms = (time.time() - t_ret_start) * 1000

        # 5. Generate Answer
        t_gen_start = time.time()
        try:
            if request.generation_mode == "llm":
                gen_resp = self.llm_generator.generate(transcript, chunks)
            else:
                gen_resp = self.extractive_generator.generate(transcript, chunks)
        except Exception as e:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=(time.time() - t_gen_start) * 1000,
                total_latency_ms=(time.time() - t_total_start) * 1000,
                error=f"Generation failed: {str(e)}"
            )
        generation_latency_ms = (time.time() - t_gen_start) * 1000
        
        # 6. Response
        return VoiceRAGResponse(
            success=True,
            transcript=transcript,
            answer=gen_resp.answer,
            source_chunk_ids=gen_resp.source_chunk_ids,
            language=stt_resp.detected_language,
            stt_latency_ms=stt_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=(time.time() - t_total_start) * 1000
        )
