import time
import logging
from typing import Optional

from .models import VoiceRAGRequest, VoiceRAGResponse
from app.stt.service import STTService
from app.stt.models import SpeechToTextRequest
from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator
from app.generation.generator import AnswerGenerator
from app.orchestration.safety_filter import check_safety

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
        
        # PyTorch and Embeddings will be lazy-loaded to bypass Render's strict 512MB startup limits
        self._embedding_service_instance = None
        
        # Store provided retrievers (e.g. from tests)
        self._retrievers = {}
        if en_retriever: self._retrievers["en"] = en_retriever
        if hi_retriever: self._retrievers["hi"] = hi_retriever
        if ta_retriever: self._retrievers["ta"] = ta_retriever
        if te_retriever: self._retrievers["te"] = te_retriever
        if ml_retriever: self._retrievers["ml"] = ml_retriever
        
        # Store provided generator or lazy load
        self._provided_extractive_generator = extractive_generator
            
        self.llm_generator = llm_generator or AnswerGenerator()
        
    @property
    def en_retriever(self): return self._retrievers.get("en")
    @property
    def hi_retriever(self): return self._retrievers.get("hi")
    @property
    def ta_retriever(self): return self._retrievers.get("ta")
    @property
    def te_retriever(self): return self._retrievers.get("te")
    @property
    def ml_retriever(self): return self._retrievers.get("ml")

    @property
    def embedding_service(self):
        if self._embedding_service_instance is None:
            from app.retrieval.embeddings import EmbeddingService
            self._embedding_service_instance = EmbeddingService()
        return self._embedding_service_instance
        
    @property
    def extractive_generator(self):
        if self._provided_extractive_generator is not None:
            return self._provided_extractive_generator
        if not hasattr(self, '_lazy_extractive_generator'):
            from app.generation.extractive_generator import ExtractiveAnswerGenerator
            self._lazy_extractive_generator = ExtractiveAnswerGenerator(embedding_service=self.embedding_service)
        return self._lazy_extractive_generator

    def execute(self, request: VoiceRAGRequest) -> VoiceRAGResponse:
        t_total_start = time.perf_counter()
        
        # 1. Validate Audio
        if not request.audio_data or len(request.audio_data) == 0:
            return VoiceRAGResponse(
                success=False,
                transcript="",
                answer="",
                source_chunk_ids=[],
                error="Empty audio data provided",
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000
            )

        # 2. STT
        stt_req = SpeechToTextRequest(
            audio_data=request.audio_data,
            language_hint=request.language_hint
        )
        
        t_stt_start = time.perf_counter()
        stt_resp = self.stt_service.transcribe(stt_req)
        stt_latency_ms = (time.perf_counter() - t_stt_start) * 1000
        
        if not stt_resp.success:
            return VoiceRAGResponse(
                success=False,
                transcript="",
                answer="",
                source_chunk_ids=[],
                stt_latency_ms=stt_latency_ms,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error=f"STT failed: {stt_resp.error}"
            )
            
        # 3. Validate Transcript (Guardrail: empty/off-topic)
        t_guard_start = time.perf_counter()
        t_rag_start = t_guard_start
        transcript = stt_resp.transcript.strip()
        if not transcript or len(transcript) < 2:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="I couldn't hear what you said.",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                guardrails_latency_ms=(time.perf_counter() - t_guard_start) * 1000,
                total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error="Empty or extremely short transcript",
                refusal_reason="Input rejected: transcript too short."
            )
        
        # Guardrail: excessively long inputs likely an attack or gibberish
        if len(transcript) > 500:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript[:500] + "...",
                answer="I'm sorry, your query is too long for me to process safely.",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                guardrails_latency_ms=(time.perf_counter() - t_guard_start) * 1000,
                total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error="Transcript too long",
                refusal_reason=f"Input rejected: transcript length {len(transcript)} exceeds 500 characters."
            )
        guardrails_latency_ms = (time.perf_counter() - t_guard_start) * 1000

        # Guardrail: Safety-intent filter (deterministic regex, O(1))
        # Runs AFTER length checks and BEFORE any retrieval.
        safety = check_safety(transcript)
        guardrails_latency_ms += safety.latency_ms  # Add to guardrail bucket
        if safety.is_unsafe:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="I'm not able to help with that request.",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                guardrails_latency_ms=guardrails_latency_ms,
                total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error="Unsafe request detected",
                refusal_reason=safety.refusal_reason
            )

        # Choose language index based on hint or detected. Default to English for simplicity if unknown.
        lang = request.language_hint or stt_resp.detected_language or "en"
        lang = lang.lower()
        # Lazy load retriever to avoid memory spikes
        def get_retriever(lang_key, index_path):
            if lang_key not in self._retrievers:
                self._retrievers[lang_key] = Retriever(index_dir=index_path, embedding_service=self.embedding_service)
                
                # Lazy-load sentence embeddings for extractive generator
                if request.generation_mode != "llm":
                    import os
                    pkl_path = os.path.join(index_path, "sentence_embeddings.pkl")
                    self.extractive_generator.load_precomputed_embeddings(pkl_path)
                    
            return self._retrievers[lang_key]

        if "hi" in lang:
            retriever = get_retriever("hi", "data/indexes/hindi")
        elif "ta" in lang:
            retriever = get_retriever("ta", "data/indexes/tamil")
        elif "te" in lang:
            retriever = get_retriever("te", "data/indexes/telugu")
        elif "ml" in lang:
            retriever = get_retriever("ml", "data/indexes/malayalam")
        else:
            retriever = get_retriever("en", "data/indexes/english")
            
        if not retriever:
            return VoiceRAGResponse(
                success=False,
                transcript=transcript,
                answer="",
                source_chunk_ids=[],
                language=stt_resp.detected_language,
                stt_latency_ms=stt_latency_ms,
                guardrails_latency_ms=guardrails_latency_ms,
                total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error="Retriever not available for language"
            )

        # 4. Retrieval
        t_ret_start = time.perf_counter()
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
                guardrails_latency_ms=guardrails_latency_ms,
                retrieval_latency_ms=(time.perf_counter() - t_ret_start) * 1000,
                total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error=f"Retrieval failed: {str(e)}"
            )
        retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000
        r_timings = getattr(retriever, "last_timings", {})

        # 5. Generate Answer
        t_gen_start = time.perf_counter()
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
                guardrails_latency_ms=guardrails_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                embedding_latency_ms=r_timings.get("embedding", 0.0),
                faiss_latency_ms=r_timings.get("faiss", 0.0),
                bm25_latency_ms=r_timings.get("bm25", 0.0),
                fusion_latency_ms=r_timings.get("fusion", 0.0),
                generation_latency_ms=(time.perf_counter() - t_gen_start) * 1000,
                total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
                total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
                error=f"Generation failed: {str(e)}"
            )
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000
        
        g_timings = getattr(self.extractive_generator, "last_timings", {}) if request.generation_mode != "llm" else {}

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
            total_rag_latency_ms=(time.perf_counter() - t_rag_start) * 1000,
            total_latency_ms=(time.perf_counter() - t_total_start) * 1000,
            guardrails_latency_ms=guardrails_latency_ms,
            embedding_latency_ms=r_timings.get("embedding", 0.0),
            faiss_latency_ms=r_timings.get("faiss", 0.0),
            bm25_latency_ms=r_timings.get("bm25", 0.0),
            fusion_latency_ms=r_timings.get("fusion", 0.0),
            grounding_latency_ms=g_timings.get("grounding", 0.0),
            refusal_reason=getattr(gen_resp, "refusal_reason", None)
        )
