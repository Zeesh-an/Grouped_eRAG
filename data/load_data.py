import gzip
import json

def load_partial_jsonl_gz(filepath, max_examples=100):
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data.append(json.loads(line))
    return data

# Example usage:
nq_file = '/local/scratch3/zmemon/Grouped_eRAG/data/simplified-nq-train.jsonl.gz'
nq_data = load_partial_jsonl_gz(nq_file, max_examples=100)

print(f"Loaded {len(nq_data)} examples")
print(json.dumps(nq_data[0], indent=2))
