import time
import pandas as pd
from retrieval import Retriever
from generator import RAGGenerator

# Subset of test questions covering all categories and some out-of-scope
TEST_QUERIES = [
    {"query": "Who is the point of contact for the MITA program?", "category": "contacts", "expected_concept": "carmen.nieves@business.rutgers.edu"},
    {"query": "What Supply Chain events are happening this week?", "category": "events", "expected_concept": "Events"},
    {"query": "How many credits do I need to minor in Finance?", "category": "requirements", "expected_concept": "Finance (29:390:329) are: 21:640:211"}, 
    {"query": "What student organizations are available for supply chain students?", "category": "student_life", "expected_concept": "Business Association of Supply Expertise (BASE)"},
    {"query": "What is the capital of France?", "category": "out_of_scope", "expected_concept": "I don't have information"},
]

def run_evaluation():
    print("Loading architecture...")
    retriever = Retriever()
    generator = RAGGenerator()
    
    results = []
    
    for item in TEST_QUERIES:
        query = item["query"]
        category = item["category"]
        expected = item["expected_concept"].lower()
        
        start_time = time.time()
        
        # Test 1: Dense Only
        distances, dense_indices = retriever.faiss_index.search(retriever.model.encode([query]), 5)
        dense_chunks = [retriever.chunk_data[i] for i in dense_indices[0]]
        dense_hit = any(expected in c["text"].lower() for c in dense_chunks)
        
        # Test 2: Hybrid (FAISS + BM25)
        hybrid_chunks, intent = retriever.retrieve(query, top_k=5, router_override=True)
        hybrid_hit = any(expected in c["text"].lower() for c in hybrid_chunks)
        
        # Test 3: LLM Answer
        answer = generator.generate_answer(query, hybrid_chunks)
        hallucination_check = "i don't have information" in answer.lower()
        
        latency = time.time() - start_time
        
        results.append({
            "Query": query,
            "Category": category,
            "Has Eventual Hit (Hybrid)": hybrid_hit,
            "Dense Hit": dense_hit,
            "Intent Routed": intent,
            "Latency (s)": round(latency, 2),
            "LLM Output Snippet (first 40 chars)": answer.replace('\n', ' ')[:40]
        })

    df = pd.DataFrame(results)
    print("\n--- EVALUATION BENCHMARK RESULTS ---")
    print(df.to_markdown(index=False))
    
    print("\nRetrieval Tradeoff Notes:")
    print("Hybrid typically matches or beats Dense because exact matching (e.g. 'BASE', emails) works better with BM25.")
    
if __name__ == "__main__":
    run_evaluation()
