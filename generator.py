from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TextGenerator:
    def __init__(self, model_name='google/flan-t5-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, inputs):
        outputs = dict()
        for query, docs in inputs.items():
            context = docs[0]  # single doc
            input_text = f"question: {query} context: {context}"
            inputs_enc = self.tokenizer([input_text], return_tensors="pt", truncation=True)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs_enc)
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            outputs[query] = answer
        return outputs