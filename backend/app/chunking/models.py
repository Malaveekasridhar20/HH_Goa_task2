from pydantic import BaseModel, Field
from typing import Optional

class Passage(BaseModel):
    query_id: str
    query_type: str
    source_lang: str
    target_lang: str
    eng_query: str
    query: str
    passage_index: int
    is_selected: bool
    text: str
    
class Chunk(BaseModel):
    chunk_id: str
    text: str
    query_id: str
    passage_index: int
    chunk_index: int
    is_selected: bool
    source_lang: str
    target_lang: str
    query_type: str
    strategy: str
    strategy_reason: str
    start_position: int
    end_position: int
