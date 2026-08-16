from pydantic import BaseModel, Field
from typing import List

class GenerationResponse(BaseModel):
    """
    Structured response from the Answer Generator.
    Ensures that source attribution is properly isolated from the text.
    """
    answer: str = Field(..., description="The grounded answer generated from the context.")
    source_chunk_ids: List[str] = Field(default_factory=list, description="List of chunk IDs that the answer was grounded on.")
    model: str = Field(..., description="The name of the generation model used.")
    generation_latency: float = Field(0.0, description="Latency in seconds for generation.")
    refusal_reason: str = Field(default="", description="Structured reason if the generation was safely refused.")
