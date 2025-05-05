import json
from erag import eval
from generator import TextGenerator
from downstream_metric import exact_match_score
from scipy.stats import kendalltau, spearmanr

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Load data
    retrieval_results = load_json('data/retrieved_docs.json')
    expected_outputs = load_json('data/nq_val.json')
    downstream_scores = load_json('data/full_rag_outputs.json')  # Query → float (e.g., EM)

    # Run eRAG evaluation
    text_generator = TextGenerator()
    retrieval_metrics = {"P_5", "MAP", "NDCG"}

    results = eval(
        retrieval_results=retrieval_results,
        expected_outputs=expected_outputs,
        text_generator=text_generator,
        downstream_metric=exact_match_score,
        retrieval_metrics=retrieval_metrics
    )

    print("\n=== Aggregated eRAG Scores ===")
    for metric, val in results['aggregated'].items():
        print(f"{metric}: {val:.4f}")

    # Compute per-query eRAG score vector (e.g., using P_5)
    erag_scores = {query: val['P_5'] for query, val in results['per_input'].items()}

    # Ensure alignment of keys
    common_queries = list(set(erag_scores) & set(downstream_scores))

    erag_vec = [erag_scores[q] for q in common_queries]
    downstream_vec = [downstream_scores[q] for q in common_queries]

    # Compute correlation
    tau, _ = kendalltau(erag_vec, downstream_vec)
    rho, _ = spearmanr(erag_vec, downstream_vec)

    print("\n=== Correlation with End-to-End RAG Performance ===")
    print(f"Kendall's Tau: {tau:.4f}")
    print(f"Spearman's Rho: {rho:.4f}")