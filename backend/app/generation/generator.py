import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import OpenAI

from app.retrieval.models import RetrievalResult
from .models import GenerationResponse
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class Provider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generates an answer based on the system and user prompts.
        Should return a dictionary containing the raw response and latency metadata.
        """
        pass

class HuggingFaceLocalProvider(Provider):
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = os.getenv("GENERATION_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        
        logger.info(f"Loading local HuggingFace model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )
        logger.info("Local model loaded.")

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        import torch
        t0 = time.time()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # The model is prompted to output JSON. Sometimes it outputs markdown blocks like ```json ... ```
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Check if it parses
            try:
                json.loads(content)
            except:
                content = json.dumps({
                    "answer": content,
                    "source_chunk_ids": []
                })
                
        except Exception as e:
            logger.error(f"Generation Error: {str(e)}")
            content = json.dumps({
                "answer": "I don't have enough information in the retrieved context to answer that.",
                "source_chunk_ids": []
            })
            
        latency = time.time() - t0
        
        return {
            "content": content,
            "latency": latency,
            "model": self.model_name
        }

class OpenAICompatibleProvider(Provider):
    def __init__(self):
        # Default to Groq's fast Llama 3 8B if available, otherwise assume a local Ollama at localhost:11434
        self.base_url = os.getenv("GENERATION_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("GENERATION_API_KEY", "ollama")  # Ollama doesn't require a real key
        self.model_name = os.getenv("GENERATION_MODEL", "qwen2.5:1.5b")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        t0 = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=256
            )
            content = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generation API Error: {str(e)}")
            # Fallback behavior on failure
            content = json.dumps({
                "answer": "I don't have enough information in the retrieved context to answer that.",
                "source_chunk_ids": []
            })
            
        latency = time.time() - t0
        
        return {
            "content": content,
            "latency": latency,
            "model": self.model_name
        }

class AnswerGenerator:
    def __init__(self, provider: Optional[Provider] = None):
        if provider is None:
            if os.getenv("GENERATION_API_KEY") or os.getenv("GENERATION_PROVIDER") == "openai_compatible":
                self.provider = OpenAICompatibleProvider()
            else:
                self.provider = HuggingFaceLocalProvider()
        else:
            self.provider = provider
            
        self.top_k = int(os.getenv("GENERATION_TOP_K", "5"))

    def _format_context(self, retrieved_chunks: List[RetrievalResult]) -> str:
        """Formats the retrieved chunks into a clear string for the LLM."""
        if not retrieved_chunks:
            return ""
            
        formatted_sources = []
        for chunk in retrieved_chunks[:self.top_k]:
            source_text = f"[Source ID: {chunk.chunk_id}]\n{chunk.text}"
            formatted_sources.append(source_text)
            
        return "\n\n".join(formatted_sources)

    def generate(self, query: str, retrieved_chunks: List[RetrievalResult], language: str = None) -> GenerationResponse:
        """
        Generates a grounded answer based strictly on the retrieved chunks.
        """
        if not retrieved_chunks:
            return GenerationResponse(
                answer="I don't have enough information in the retrieved context to answer that.",
                source_chunk_ids=[],
                model=self.provider.model_name if hasattr(self.provider, 'model_name') else "unknown",
                generation_latency=0.0
            )

        # 1. Construct Context
        context_str = self._format_context(retrieved_chunks)
        
        # 2. Construct Prompt
        user_prompt = f"Source Context:\n{context_str}\n\nUser Query: {query}"
        
        # 3. Generate
        result = self.provider.generate(SYSTEM_PROMPT, user_prompt)
        
        # 4. Parse Response safely
        try:
            parsed = json.loads(result["content"])
            answer = parsed.get("answer", "I don't have enough information in the retrieved context to answer that.")
            source_ids = parsed.get("source_chunk_ids", [])
            if not isinstance(source_ids, list):
                source_ids = []
        except Exception as e:
            logger.error(f"JSON Parsing Error in Generation: {str(e)}")
            answer = "I don't have enough information in the retrieved context to answer that."
            source_ids = []
            
        return GenerationResponse(
            answer=answer,
            source_chunk_ids=source_ids,
            model=result.get("model", "unknown"),
            generation_latency=result.get("latency", 0.0)
        )
