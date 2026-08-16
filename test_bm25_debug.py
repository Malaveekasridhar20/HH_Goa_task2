from app.retrieval.bm25_index import BM25Index
b=BM25Index()
docs=['The quick brown fox jumps over the lazy dog.', 'A quick brown dog outpaces a fast fox.', '\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e', 'Hello world in Hindi is \u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e']
b.build_index(docs)
print(b.tokenize('\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e'))
print(b.tokenize('\u0928\u092e\u0938\u094d\u0924\u0947'))
print(b.bm25.get_scores(b.tokenize('\u0928\u092e\u0938\u094d\u0924\u0947')))
