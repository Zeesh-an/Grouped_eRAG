import json
import gzip
import re
from typing import List
import wikipedia
from tqdm import tqdm  # Add this import

def clean_html(raw_text: str) -> str:
    return re.sub(r'<[^>]+>', '', raw_text)

def extract_short_answers(example) -> List[str]:
    if example.get('annotations'):
        ann = example['annotations'][0]
        return [' '.join(example['document_text'].split()[sa['start_token']:sa['end_token']])
                for sa in ann.get('short_answers', [])]
    return []

def extract_long_answer(example) -> str:
    if example.get('annotations'):
        ann = example['annotations'][0]
        long = ann['long_answer']
        if long['start_token'] != -1 and long['end_token'] != -1:
            return ' '.join(example['document_text'].split()[long['start_token']:long['end_token']])
    return ""

def wikipedia_search(query: str, num_docs: int = 20) -> List[str]:
    try:
        titles = wikipedia.search(query, results=num_docs)
        docs = []
        for title in titles:
            try:
                summary = wikipedia.summary(title, auto_suggest=False)
                docs.append(summary)
            except Exception:
                continue  # Skip problematic pages
        return docs
    except Exception:
        return []

def process_nq_with_wikipedia(nq_path, out_retrieved_docs, out_nq_val, max_examples=100, top_k=20):
    data = []
    with gzip.open(nq_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data.append(json.loads(line))

    retrieved_docs = {}
    nq_val = {}

    for ex in tqdm(data, desc="Processing examples"):  # Use tqdm here
        query = ex['question_text'].strip()
        retrieved = wikipedia_search(query, num_docs=top_k)
        retrieved_docs[query] = retrieved

        answers = extract_short_answers(ex)
        if not answers:
            long = extract_long_answer(ex)
            if long:
                answers = [long]
        if answers:
            nq_val[query] = answers

    # --- Compute average number of retrieved docs ---
    total_retrieved = sum(len(docs) for docs in retrieved_docs.values())
    avg_retrieved_per_query = total_retrieved / len(retrieved_docs) if retrieved_docs else 0
    print(f"Average number of Wikipedia articles retrieved per query: {avg_retrieved_per_query:.2f}")

    with open(out_retrieved_docs, 'w') as f:
        json.dump(retrieved_docs, f, indent=2)

    with open(out_nq_val, 'w') as f:
        json.dump(nq_val, f, indent=2)

# Example usage
process_nq_with_wikipedia(
    nq_path="/local/scratch3/zmemon/Grouped_eRAG/data/simplified-nq-train.jsonl.gz",
    out_retrieved_docs="retrieved_docs.json",
    out_nq_val="nq_val.json"
)
