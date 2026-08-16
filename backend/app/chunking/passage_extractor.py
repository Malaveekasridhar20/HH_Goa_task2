import re
from typing import List, Dict, Any
from app.chunking.models import Passage

class PassageExtractor:
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Conservative text normalization:
        - trim leading/trailing whitespace
        - normalize repeated spaces
        - normalize unnecessary repeated line breaks
        """
        if not text:
            return ""
        # Replace multiple spaces/tabs with a single space (leaving newlines intact)
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace 3 or more line breaks with just 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def extract_passages(record: Dict[str, Any], extract_english: bool = False) -> List[Passage]:
        passages = []
        
        query_id = str(record.get('query_id', ''))
        query_type = str(record.get('query_type', ''))
        source_lang = str(record.get('source_lang', ''))
        target_lang = str(record.get('target_lang', ''))
        eng_query = str(record.get('Eng_Query', ''))
        query = str(record.get('query', ''))
        
        passages_obj = record.get('passages', {})
        if not isinstance(passages_obj, dict):
            passages_obj = {}
            
        passage_texts = []
        if extract_english and 'English_passages' in passages_obj:
            passage_texts = passages_obj['English_passages']
        elif not extract_english and 'Translated_passages' in passages_obj:
            passage_texts = passages_obj['Translated_passages']
        elif 'passage_text' in passages_obj:
            passage_texts = passages_obj['passage_text']
        elif 'English_passages' in passages_obj:
            passage_texts = passages_obj['English_passages']
            
        is_selected_list = passages_obj.get('is_selected', [])
        
        for idx, text_raw in enumerate(passage_texts):
            if text_raw is None:
                continue
            text = PassageExtractor.normalize_text(str(text_raw))
            if not text:
                continue
                
            is_selected = False
            if idx < len(is_selected_list):
                # Ensure it's evaluated properly (e.g., 1 -> True, 0 -> False)
                is_selected = bool(is_selected_list[idx])
                
            passages.append(Passage(
                query_id=query_id,
                query_type=query_type,
                source_lang=source_lang,
                target_lang=target_lang,
                eng_query=eng_query,
                query=query,
                passage_index=idx,
                is_selected=is_selected,
                text=text
            ))
            
        return passages
