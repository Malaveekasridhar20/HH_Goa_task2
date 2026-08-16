from app.retrieval.bm25_index import BM25Index
b=BM25Index()
docs=['The quick brown fox jumps over the lazy dog.', 'A quick brown dog outpaces a fast fox.', 'namaste duniya', 'Hello world in Hindi is namaste duniya']
b.build_index(docs)
print(b.bm25.get_scores(b.tokenize('namaste')))
