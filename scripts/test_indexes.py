import sys
sys.path.insert(0, 'backend')
from app.retrieval.retriever import Retriever

langs = {
    'english': 'data/indexes/english',
    'hindi':   'data/indexes/hindi',
    'tamil':   'data/indexes/tamil',
    'telugu':  'data/indexes/telugu',
    'malayalam': 'data/indexes/malayalam',
}

queries = {
    'english':   'What is the capital of France?',
    'hindi':     'भारत की राजधानी क्या है?',
    'tamil':     'இந்தியாவின் தலைநகர் எது?',
    'telugu':    'భారతదేశ రాజధాని ఏది?',
    'malayalam': 'ഇന്ത്യയുടെ തലസ്ഥാനം ഏതാണ്?',
}

for lang, idx_dir in langs.items():
    try:
        r = Retriever(index_dir=idx_dir)
        results = r.retrieve_vector(queries[lang], top_k=3)
        if results:
            score = getattr(results[0], 'score', 'N/A')
            print(f'{lang}: PASS - retrieved {len(results)} chunks, first_score={score}')
        else:
            print(f'{lang}: FAIL - no results returned')
    except Exception as e:
        print(f'{lang}: FAIL - {e}')
