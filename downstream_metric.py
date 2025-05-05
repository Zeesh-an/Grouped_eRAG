def exact_match_score(predictions, golds):
    def normalize(ans): return ans.lower().strip()
    scores = {}
    for q, pred in predictions.items():
        true_answers = golds[q]
        scores[q] = int(any(normalize(pred) == normalize(ans) for ans in true_answers))
    return scores