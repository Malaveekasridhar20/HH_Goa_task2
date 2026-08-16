import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'backend')
from dotenv import load_dotenv
load_dotenv('backend/.env')
from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

# Test te_2 grounding
r_te = Retriever(index_dir='data/indexes/telugu')
gen = ExtractiveAnswerGenerator(embedding_service=r_te.embedding_service)

query_te2 = 'లక్ష్మీ నాగమాధిరి. ఆంధ్రప్రదేశ్లో కరోనా వైరస్'
chunks_te2 = r_te.retrieve_vector(query_te2, top_k=5)
context_te2 = ' '.join([c.text for c in chunks_te2])
resp_te2 = gen.generate(query_te2, chunks_te2)
print('te_2 answer exists:', bool(resp_te2.answer))
print('te_2 score:', resp_te2.score if hasattr(resp_te2, 'score') else 'N/A')
answer_in_ctx = resp_te2.answer in context_te2 if resp_te2.answer else False
print('te_2 answer in context:', answer_in_ctx)

# ml_3 - 'Main news'
r_ml = Retriever(index_dir='data/indexes/malayalam')
gen_ml = ExtractiveAnswerGenerator(embedding_service=r_ml.embedding_service)
query_ml3 = 'Main news'
chunks_ml3 = r_ml.retrieve_vector(query_ml3, top_k=5)
resp_ml3 = gen_ml.generate(query_ml3, chunks_ml3)
print('ml_3 answer exists:', bool(resp_ml3.answer))
print('ml_3 score:', resp_ml3.score if hasattr(resp_ml3, 'score') else 'N/A')
context_ml3 = ' '.join([c.text for c in chunks_ml3])
answer_in_ctx_ml = resp_ml3.answer in context_ml3 if resp_ml3.answer else False
print('ml_3 answer in context:', answer_in_ctx_ml)

# Check what fields exist in generation response
print('Generation response fields:', dir(resp_te2))
