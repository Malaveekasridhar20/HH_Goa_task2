import sys
sys.path.append("backend")

from app.retrieval.retriever import Retriever
from app.generation.extractive_generator import ExtractiveAnswerGenerator

def diagnose():
    with open("diagnose_output.txt", "w", encoding="utf-8") as f:
        def print_f(*args, **kwargs):
            print(*args, file=f, **kwargs)
        
        print_f("Loading Hindi Retriever...")
        hi_retriever = Retriever(index_dir="data/indexes/hindi")
        ext_gen = ExtractiveAnswerGenerator(embedding_service=hi_retriever.embedding_service)

        query = "भारत की राजधानी क्या है?"
        print_f(f"\nQuery: {query}")
    
        chunks = hi_retriever.retrieve_vector(query, top_k=5)
        
        print_f("\n--- FAISS RETRIEVED CHUNKS ---")
        for i, chunk in enumerate(chunks):
            print_f(f"[{i+1}] ID: {chunk.chunk_id}")
            print_f(f"    Score: {chunk.score:.4f}")
            print_f(f"    Text: {chunk.text}")
            
        print_f("\n--- EXTRACTIVE GENERATOR ---")
        print_f(f"Configured threshold: {ext_gen.relevance_threshold}")
    
        gen_resp = ext_gen.generate(query, chunks)
        print_f(f"\nFinal Answer: {gen_resp.answer}")
        
        if not chunks:
            print_f("No chunks to score.")
        else:
            query_emb = ext_gen.embedding_service.encode_query(query)
            all_sentences = []
            import re
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            for c in chunks:
                sents = re.split(r'(?<=[।?!|.])\s+', c.text.strip())
                for s in sents:
                    if len(s.strip()) > 5:
                        all_sentences.append((s.strip(), c.chunk_id))
            
            if all_sentences:
                s_texts = [s[0] for s in all_sentences]
                s_embs = ext_gen.embedding_service.encode_documents(s_texts)
                sims = cosine_similarity([query_emb], s_embs)[0]
                best_idx = np.argmax(sims)
                best_score = float(sims[best_idx])
                best_sent = s_texts[best_idx]
                print_f(f"Best sentence score: {best_score:.4f}")
                print_f(f"Best sentence: {best_sent}")
                
        print_f("\n\nLoading English Retriever for known-good query...")
        en_retriever = Retriever(index_dir="data/indexes/english")
        ext_gen_en = ExtractiveAnswerGenerator(embedding_service=en_retriever.embedding_service)
        
        good_query = "What is a corporation?"
        chunks_en = en_retriever.retrieve_vector(good_query, top_k=5)
        gen_resp_en = ext_gen_en.generate(good_query, chunks_en)
        
        print_f(f"\nKnown-Good Query: {good_query}")
        print_f(f"Answer: {gen_resp_en.answer}")
        print_f(f"Grounded: {'PASS' if gen_resp_en.source_chunk_ids else 'FAIL'}")

if __name__ == "__main__":
    diagnose()
