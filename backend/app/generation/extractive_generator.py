import os
import re
import time
import threading
import numpy as np
import logging
from collections import OrderedDict
from typing import List, Optional

from app.retrieval.models import RetrievalResult
from app.generation.models import GenerationResponse
from app.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-Embedding LRU Cache
# ---------------------------------------------------------------------------
# Keyed on (chunk_id, sentence_index, sentence_text) — fully deterministic
# and immutable. chunk_id ties the entry to the indexed document; sentence_text
# acts as a content guard so any hypothetical text change auto-invalidates.
#
# Bound: 50 000 sentence slots ≈ ~75 MB worst-case (384-dim float32).
# Eviction: LRU (OrderedDict move_to_end / popitem).
# Thread safety: lock is held only for O(1) lookup/insert, never during
# the (potentially slow) encode_documents() call.
# ---------------------------------------------------------------------------

_CACHE_MAX_ENTRIES = 50_000


class _SentenceEmbeddingCache:
    """Thread-safe bounded LRU cache for sentence embeddings."""

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES):
        self._max = max_entries
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key, value):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._max:
                    self._store.popitem(last=False)
                self._store[key] = value

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return {'hits': self.hits, 'misses': self.misses,
                'total': total, 'hit_rate': round(rate, 4)}


class ExtractiveAnswerGenerator:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.relevance_threshold = float(
            os.getenv("EXTRACTIVE_RELEVANCE_THRESHOLD", "0.85")
        )
        # Per-instance cache — naturally partitioned by language since each
        # language pipeline instantiates its own generator.
        self._cache = _SentenceEmbeddingCache()

    def _split_sentences(self, text: str) -> List[str]:
        """Splits text into sentences based on ., ?, !, and Indic danda ।"""
        sentences = re.split(r'([.?!।]+)', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i].strip() + sentences[i + 1].strip()
            if sent:
                result.append(sent)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        return result

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        language: str = None,
    ) -> GenerationResponse:
        t0 = time.time()

        if not retrieved_chunks:
            return GenerationResponse(
                answer="I don't have enough information in the retrieved context to answer that.",
                source_chunk_ids=[],
                model="extractive",
                generation_latency=time.time() - t0,
            )

        # ── Step 1: Collect all candidate sentences and their cache keys ──
        candidate_sentences = []
        sentence_to_chunk_id = []
        cache_keys = []          # one key per candidate sentence

        for chunk in retrieved_chunks:
            sentences = self._split_sentences(chunk.text)
            for idx, s in enumerate(sentences):
                s = s.strip()
                if len(s) > 5:
                    candidate_sentences.append(s)
                    sentence_to_chunk_id.append(chunk.chunk_id)
                    cache_keys.append((chunk.chunk_id, idx, s))

        if not candidate_sentences:
            return GenerationResponse(
                answer="I don't have enough information in the retrieved context to answer that.",
                source_chunk_ids=[],
                model="extractive",
                generation_latency=time.time() - t0,
            )

        # ── Step 2: Cache lookup — collect which sentences are missing ──
        # result_embs[i] = embedding or None
        result_embs: List[Optional[np.ndarray]] = [None] * len(candidate_sentences)
        missing_positions = []   # indices into candidate_sentences

        for i, key in enumerate(cache_keys):
            cached = self._cache.get(key)
            if cached is not None:
                result_embs[i] = cached
            else:
                missing_positions.append(i)

        # ── Step 3: Batch-encode ALL missing sentences in ONE call ──
        # This preserves the original single-batch encode_documents() pattern,
        # so cold-path performance is identical to the un-cached version.
        if missing_positions:
            missing_texts = [candidate_sentences[i] for i in missing_positions]
            new_embs = self.embedding_service.encode_documents(missing_texts)
            for list_pos, orig_idx in enumerate(missing_positions):
                emb = new_embs[list_pos]
                self._cache.put(cache_keys[orig_idx], emb)
                result_embs[orig_idx] = emb

        # ── Step 4: Assemble final embedding matrix ──
        sentence_embs = np.array(result_embs)

        # ── Step 5: Query embedding + cosine similarity (unchanged) ──
        query_emb = self.embedding_service.encode_query(query)
        scores = np.dot(sentence_embs, query_emb)

        # ── Step 6: Best-sentence selection + refusal threshold (unchanged) ──
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < self.relevance_threshold:
            return GenerationResponse(
                answer="I don't have enough information in the retrieved context to answer that.",
                source_chunk_ids=[],
                model="extractive",
                generation_latency=time.time() - t0,
            )

        best_sentence = candidate_sentences[best_idx]
        best_chunk_id = sentence_to_chunk_id[best_idx]

        return GenerationResponse(
            answer=best_sentence,
            source_chunk_ids=[best_chunk_id],
            model="extractive",
            generation_latency=time.time() - t0,
        )

    def cache_stats(self) -> dict:
        """Expose cache statistics for monitoring and benchmarking."""
        return self._cache.stats()
