# import json
# import numpy as np
# from scipy.stats import spearmanr, kendalltau
# from typing import Dict, List
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# from tqdm import tqdm
# import pandas as pd

# # ----------------------------
# # Load retrieved docs and gold answers
# # ----------------------------
# with open("data/retrieved_docs.json", "r") as f:
#     retrieved_docs: Dict[str, List[str]] = json.load(f)

# with open("data/nq_val.json", "r") as f:
#     gold_answers: Dict[str, List[str]] = json.load(f)

# # ----------------------------
# # Load FiD Model (Intel's flan-t5-base fine-tuned on NQ)
# # ----------------------------
# tokenizer = AutoTokenizer.from_pretrained("Intel/fid_flan_t5_base_nq")
# model = AutoModelForSeq2SeqLM.from_pretrained("Intel/fid_flan_t5_base_nq")
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # ----------------------------
# # Answer generation using FiD
# # ----------------------------
# def generate_fid_answer(query: str, passages: List[str]) -> str:
#     inputs = [f"question: {query} context: {p}" for p in passages]
#     tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
#     with torch.no_grad():
#         output = model.generate(**tokenized, max_new_tokens=32)
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()

# # ----------------------------
# # Exact Match scoring (EM), as in eRAG paper for NQ
# # ----------------------------
# def exact_match_score(pred: str, golds: List[str]) -> float:
#     pred = pred.strip().lower()
#     return float(any(pred == gold.strip().lower() for gold in golds))

# # ----------------------------
# # eRAG Scoring Loop
# # ----------------------------
# erag_scores = {}
# downstream_em_scores = {}  # To compare with query-level RAG performance

# for query, docs in tqdm(retrieved_docs.items(), desc="Scoring queries"):
#     if query not in gold_answers:
#         continue
#     doc_scores = []
#     topk_generated = []
#     for doc in docs:
#         answer = generate_fid_answer(query, [doc])  # Single doc per call
#         score = exact_match_score(answer, gold_answers[query])
#         doc_scores.append(score)
#         topk_generated.append(answer)
#     erag_scores[query] = doc_scores

#     # Approximate downstream performance: use top-k docs (all) to generate answer
#     full_answer = generate_fid_answer(query, docs)
#     downstream_em_scores[query] = exact_match_score(full_answer, gold_answers[query])

# # ----------------------------
# # Rank correlation metrics (Document-level)
# # ----------------------------
# tau_vals, spearman_vals = [], []
# for query, scores in erag_scores.items():
#     ideal = sorted(scores, reverse=True)
#     if len(set(ideal)) <= 1:
#         continue
#     tau, _ = kendalltau(scores, ideal)
#     spearman, _ = spearmanr(scores, ideal)
#     if not np.isnan(tau):
#         tau_vals.append(tau)
#     if not np.isnan(spearman):
#         spearman_vals.append(spearman)

# average_tau = np.mean(tau_vals)
# average_spearman = np.mean(spearman_vals)

# # ----------------------------
# # Query-level correlation (eRAG vs downstream RAG EM)
# # ----------------------------
# erag_query_scores = [np.mean(scores) for scores in erag_scores.values()]
# downstream_em = [downstream_em_scores[q] for q in erag_scores.keys()]

# query_tau, _ = kendalltau(erag_query_scores, downstream_em)
# query_spearman, _ = spearmanr(erag_query_scores, downstream_em)

# # ----------------------------
# # Final Output
# # ----------------------------
# print("\n=== eRAG Evaluation Metrics ===")
# print(pd.DataFrame({
#     "Metric": [
#         "Average Kendall's Tau (Doc-Level)",
#         "Average Spearman (Doc-Level)",
#         "Kendall's Tau (Query-Level)",
#         "Spearman (Query-Level)"
#     ],
#     "Score": [
#         average_tau,
#         average_spearman,
#         query_tau,
#         query_spearman
#     ]
# }).to_string(index=False))


from collections import Counter
import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import os
import string
import re
os.environ["TRANSFORMERS_CACHE"] = "/local/scratch3/zmemon/Grouped_eRAG"

# ----------------------------
# Config
# ----------------------------
GROUP_SIZE = 7
ALPHA = 0.8  # Regularization weight between eRAG score and complementarity

# ----------------------------
# Load retrieved docs and gold answers
# ----------------------------
with open("data/retrieved_docs.json", "r") as f:
    retrieved_docs: Dict[str, List[str]] = json.load(f)

with open("data/nq_val.json", "r") as f:
    gold_answers: Dict[str, List[str]] = json.load(f)

# ----------------------------
# Load FiD Model and Embedder
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
def semantic_score(pred, golds):
    pred_emb = embedder.encode([pred], convert_to_tensor=True)
    gold_embs = embedder.encode(golds, convert_to_tensor=True)
    scores = torch.cosine_similarity(pred_emb, gold_embs)[0]
    return torch.max(scores).item()

def white_space_fix(text: str) -> str:
    """
    Removes extra whitespaces and trims leading/trailing spaces.
    Example: "  Hello   world!  " -> "Hello world!"
    """
    return re.sub(r'\s+', ' ', text).strip()


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)
# ----------------------------
# Generation utilities
# ----------------------------
def generate_fid_answer(query: str, passages: List[str]) -> str:
    inputs = [f"question: {query} context: {p}" for p in passages]
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(**tokenized, max_new_tokens=32)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()


def get_redundancy_score(doc_idx: int, embeddings: np.ndarray) -> float:
    sims = cosine_similarity([embeddings[doc_idx]], embeddings)[0]
    return (np.sum(sims) - 1) / (len(sims) - 1)  # Exclude self similarity

def exact_match_score(pred: str, golds: List[str]) -> float:
    pred = pred.strip().lower()
    # return float(any(pred == gold.strip().lower() for gold in golds))
    return 0.5 * (max(f1_score(pred, gold) for gold in golds)) +  0.5 * semantic_score(pred, golds)


# ----------------------------
# Complementarity scoring
# ----------------------------
def get_complementary_score(query: str, doc: str, neighbors: List[str], golds: List[str]) -> float:
    group_with = [doc] + neighbors
    group_without = neighbors

    ans_with = generate_fid_answer(query, group_with)
    ans_without = generate_fid_answer(query, group_without)

    score_with = exact_match_score(ans_with, golds)
    score_without = exact_match_score(ans_without, golds)

    return max(score_with - score_without, 0.0)

# ----------------------------
# Main scoring
# ----------------------------
erag_scores = {}
reg_erag_scores = {}
downstream_em_scores = {}
num = 0

def mmr_neighbors(doc_idx, doc_embs, query_emb, k, lambda_val=0.7):
    sims_to_query = cosine_similarity([query_emb], doc_embs)[0]
    selected = []
    candidates = list(range(len(doc_embs)))
    candidates.remove(doc_idx)  # don't select self

    while len(selected) < k and candidates:
        mmr_scores = []
        for i in candidates:
            redundancy = max([cosine_similarity([doc_embs[i]], [doc_embs[j]])[0][0] for j in selected], default=0)
            score = lambda_val * sims_to_query[i] - (1 - lambda_val) * redundancy
            mmr_scores.append((score, i))
        _, best = max(mmr_scores)
        selected.append(best)
        candidates.remove(best)

    return selected


for query, docs in tqdm(retrieved_docs.items(), desc="Scoring queries"):
    if query not in gold_answers or len(docs) < GROUP_SIZE:
        continue
    num += 1
    if num>10:
        break
    golds = gold_answers[query]
    emb = embedder.encode(docs)
    doc_scores = []
    reg_scores = []

    # for i, doc in enumerate(docs):
    #     answer = generate_fid_answer(query, [doc])
    #     # print("Answer: ", answer)
    #     print("Query and answer: ", query, answer)
    #     print("Gold answers: ", golds)
    #     score = exact_match_score(answer, golds)
    #     doc_scores.append(score)

    #     # Get top (GROUP_SIZE-1) most dissimilar docs (for complementarity)
    #     sims = cosine_similarity([emb[i]], emb)[0]
    #     sims[i] = float('inf')  # effectively removes self
    #     neighbor_ids = np.argsort(sims)[:GROUP_SIZE-1]
    #     print("Doc id itself: ", i, " Neighbor ids: ", neighbor_ids)
    #     neighbors = [docs[j] for j in neighbor_ids]

    #     comp_score = get_complementary_score(query, doc, neighbors, golds)
    #     redundancy_score = get_redundancy_score(i, emb)
    #     reg_score = ALPHA * score + (1 - ALPHA) * comp_score - 0.1 * redundancy_score  # 0.2 is γ, you can tune it
    #     print("Score and reg score: ", score, reg_score)
    #     reg_scores.append(reg_score)
    # ---------------------
    # Main scoring loop
    # ---------------------
    query_emb = embedder.encode([query])[0]
    for i, doc in enumerate(docs):
        answer = generate_fid_answer(query, [doc])
        print("Query and answer: ", query, answer)
        print("Gold answers: ", golds)
        score = exact_match_score(answer, golds)
        doc_scores.append(score)

        neighbor_ids = mmr_neighbors(i, emb, query_emb, GROUP_SIZE - 1, lambda_val=0.7)
        
        print("Doc id itself: ", i, " MMR-based Neighbor ids: ", neighbor_ids)
        neighbors = [docs[j] for j in neighbor_ids]

        comp_score = get_complementary_score(query, doc, neighbors, golds)
        redundancy_score = get_redundancy_score(i, emb)
        reg_score = ALPHA * score + (1 - ALPHA) * comp_score - 0.1 * redundancy_score  # You can tune 0.1 as γ
        print("Score and reg score: ", score, reg_score)
        reg_scores.append(reg_score)
    erag_scores[query] = doc_scores
    reg_erag_scores[query] = reg_scores

    downstream_answer = generate_fid_answer(query, docs)
    downstream_em_scores[query] = exact_match_score(downstream_answer, golds)

# ----------------------------
# Metrics
# ----------------------------
def rank_correlation(scores_dict):
    tau_vals, spearman_vals = [], []
    for scores in scores_dict.values():
        ideal = sorted(scores, reverse=True)
        if len(set(ideal)) <= 1:
            continue
        tau, _ = kendalltau(scores, ideal)
        spearman, _ = spearmanr(scores, ideal)
        if not np.isnan(tau): tau_vals.append(tau)
        if not np.isnan(spearman): spearman_vals.append(spearman)
    return np.mean(tau_vals), np.mean(spearman_vals)

# Document-level
tau_orig, sp_orig = rank_correlation(erag_scores)
tau_reg, sp_reg = rank_correlation(reg_erag_scores)

# Query-level
erag_query_scores = [np.mean(scores) for scores in erag_scores.values()]
reg_query_scores = [np.mean(scores) for scores in reg_erag_scores.values()]
downstream_ems = [downstream_em_scores[q] for q in erag_scores.keys()]

q_tau_orig, _ = kendalltau(erag_query_scores, downstream_ems)
q_tau_reg, _ = kendalltau(reg_query_scores, downstream_ems)
q_sp_orig, _ = spearmanr(erag_query_scores, downstream_ems)
q_sp_reg, _ = spearmanr(reg_query_scores, downstream_ems)

# ----------------------------
# Final report
# ----------------------------
df = pd.DataFrame({
    "Metric": [
        "Doc-Level Tau (Original)", "Doc-Level Spearman (Original)",
        "Doc-Level Tau (Regularized)", "Doc-Level Spearman (Regularized)",
        "Query-Level Tau (Original)", "Query-Level Spearman (Original)",
        "Query-Level Tau (Regularized)", "Query-Level Spearman (Regularized)"
    ],
    "Score": [
        tau_orig, sp_orig,
        tau_reg, sp_reg,
        q_tau_orig, q_sp_orig,
        q_tau_reg, q_sp_reg
    ]
})

print("\n=== Final eRAG + Complementarity Evaluation ===")
print(df.to_string(index=False))