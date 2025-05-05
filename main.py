# eRAG Reproduction Pipeline for Natural Questions (NQ)

import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
import pytrec_eval

### ------------------- Loaders and Utils -------------------
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def normalize_answer(ans):
    return ans.lower().strip()

def exact_match(pred: str, golds: List[str]) -> int:
    pred = normalize_answer(pred)
    return int(any(pred == normalize_answer(g) for g in golds))

### ------------------- Text Generator -------------------
class TextGenerator:
    def __init__(self, model_name='google/t5-small-ssm-nq'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, inputs: Dict[str, List[str]]) -> Dict[str, str]:
        outputs = dict()
        for query, docs in inputs.items():
            context = docs[0]
            input_text = f"question: {query} context: {context}"
            inputs_enc = self.tokenizer([input_text], return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs_enc, max_new_tokens=32)
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            outputs[query] = answer
        return outputs

### ------------------- Downstream Metric -------------------
def exact_match_score(predictions, golds):
    scores = {}
    for q, pred in predictions.items():
        true_answers = golds[q]
        scores[q] = exact_match(pred, true_answers)
    return scores

### ------------------- eRAG Evaluation Core -------------------
def erag_eval(retrieval_results, expected_outputs, text_generator, downstream_metric, retrieval_metrics):
    flatten_inputs = {
        f'{q}@{i}': {'query': q, 'document': [doc]} 
        for q, docs in retrieval_results.items() for i, doc in enumerate(docs)
    }
    max_len = max(len(docs) for docs in retrieval_results.values())
    evaluation_scores = {}

    for i in range(max_len):
        current_input = {}
        current_expected = {}
        for q in retrieval_results:
            key = f'{q}@{i}'
            if key in flatten_inputs:
                current_input[q] = flatten_inputs[key]['document']
                current_expected[q] = expected_outputs[q]
        generated = text_generator(current_input)
        scores = downstream_metric(generated, current_expected)
        for q, s in scores.items():
            evaluation_scores[f'{q}@{i}'] = s

    run = {q: {str(j): len(retrieval_results[q]) - j for j in range(len(retrieval_results[q]))} for q in retrieval_results}
    qrel = {q: {str(j): evaluation_scores[f'{q}@{j}'] for j in range(len(retrieval_results[q]))} for q in retrieval_results}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, retrieval_metrics)
    results = evaluator.evaluate(run)
    aggregated = {m: sum([res[m] for res in results.values()]) / len(results) for m in retrieval_metrics}
    return {'per_input': results, 'aggregated': aggregated}

### ------------------- Full RAG Output Generator -------------------
def generate_full_rag_outputs(retrieval_results, expected_outputs, model_name='google/flan-t5-small'):
    model = TextGenerator(model_name)
    output_scores = {}
    for query, docs in tqdm(retrieval_results.items(), desc="Full RAG"):
        golds = expected_outputs.get(query, [])
        try:
            answer = model({query: docs})[query]
            score = exact_match(answer, golds)
            output_scores[query] = score
        except:
            output_scores[query] = 0
    return output_scores

### ------------------- Main Pipeline -------------------
if __name__ == '__main__':
    retrieval_results = load_json('data/retrieved_docs.json')
    expected_outputs = load_json('data/nq_val.json')

    # Step 1: Run eRAG
    text_gen = TextGenerator()
    retrieval_metrics = {"P_5", "MAP", "NDCG"}
    erag_results = erag_eval(retrieval_results, expected_outputs, text_gen, exact_match_score, retrieval_metrics)
    print("\n=== Aggregated eRAG Scores ===")
    for k, v in erag_results['aggregated'].items():
        print(f"{k}: {v:.4f}")

    # Step 2: Run full RAG generation for correlation
    full_scores = generate_full_rag_outputs(retrieval_results, expected_outputs)
    save_json('data/full_rag_outputs.json', full_scores)

    erag_scores = {q: v['P_5'] for q, v in erag_results['per_input'].items()}
    common_qs = set(erag_scores) & set(full_scores)
    x = [erag_scores[q] for q in common_qs]
    y = [full_scores[q] for q in common_qs]
    tau, _ = kendalltau(x, y)
    rho, _ = spearmanr(x, y)

    print("\n=== Correlation with Full RAG ===")
    print(f"Kendall's Tau: {tau:.4f}")
    print(f"Spearman's Rho: {rho:.4f}")
