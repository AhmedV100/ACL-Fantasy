import time
import pandas as pd
from llm_manager import RAGManager

# Golden Dataset
TEST_SET = [
    {
        "query": "Analyze Manchester United attacking performance",
        "expected_keywords": ["Bruno", "Rashford", "Man Utd", "Manchester United"],
        "category": "Team Analysis"
    },
    {
        "query": "Who is the best captain pick for the next gameweek?",
        "expected_keywords": ["Haaland", "Kane", "Salah"],
        "category": "Recommendation"
    },
    {
        "query": "Compare Haaland and Kane",
        "expected_keywords": ["Haaland", "Kane", "points", "goals"],
        "category": "Comparison"
    },
    {
        "query": "What are Salah's stats?",
        "expected_keywords": ["Salah", "goals", "assists", "points"],
        "category": "Stats"
    },
    {
        "query": "Who are the top midfieders?",
        "expected_keywords": ["Saka", "Salah", "Odegaard", "Rashford"],
        "category": "Recommendation"
    }
]

def run_benchmark():
    print("🚀 Starting Benchmark Run...")
    results = []
    rag = RAGManager() # Uses default config

    for case in TEST_SET:
        query = case["query"]
        expected = case["expected_keywords"]
        category = case["category"]
        
        print(f"Processing: '{query}'...")
        
        start_time = time.time()
        response, context, queries, usage = rag.process_query(query)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Accuracy Check (Simple Keyword Match)
        hit_count = sum(1 for kw in expected if kw.lower() in response.lower())
        accuracy_score = hit_count / len(expected)
        is_correct = accuracy_score > 0.5 # Threshold

        results.append({
            "query": query,
            "category": category,
            "latency_sec": round(latency, 2),
            "input_tokens": usage['input_tokens'],
            "output_tokens": usage['output_tokens'],
            "total_tokens": usage['total_tokens'],
            "cost_usd": round(usage['cost'], 5),
            "accuracy_score": accuracy_score,
            "keywords_found": hit_count,
            "keywords_total": len(expected)
        })

    rag.close()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg_latency = df['latency_sec'].mean()
    avg_cost = df['cost_usd'].mean()
    avg_accuracy = df['accuracy_score'].mean()
    
    print("\n--- Benchmark Summary ---")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Avg Cost/Query: ${avg_cost:.5f}")
    print(f"Avg Keyword Accuracy: {avg_accuracy:.1%}")
    
    # Save
    df.to_csv("benchmark_results.csv", index=False)
    print("\n📄 Results saved to 'benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()
