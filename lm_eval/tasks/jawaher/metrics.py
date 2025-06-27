from evaluate import load
import re
from lm_eval.api.metrics import register_metric

# @register_metric(
#     metric="arabic_exact_match",
#     higher_is_better=True,
#     output_type="generate_until",
#     aggregation="mean",
# )
def arabic_exact_match(predictions, references, **kwargs):
    """
    Exact match for Arabic text with normalization
    """
    def normalize_arabic(text):
        # Remove diacritics and normalize
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)  # Remove diacritics

        # Normalize yeh variations (Persian/Urdu yeh to Arabic yeh)
        text = text.replace('ی', 'ي')  # Persian/Urdu yeh to Arabic yeh
        text = text.replace('ى', 'ي')  # Alef maksura to Arabic yeh
        
        # Normalize alef variations
        text = text.replace('أ', 'ا')  # Alef with hamza above
        text = text.replace('إ', 'ا')  # Alef with hamza below
        text = text.replace('آ', 'ا')  # Alef with madda

        # Remove common punctuation
        text = re.sub(r'[.,;:!?؟،؛]', '', text)

        text = text.strip()
        return text
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_arabic(pred)
        ref_norm = normalize_arabic(ref)
        scores.append(1.0 if pred_norm == ref_norm else 0.0)
    
    return sum(scores) / len(scores)

    
def bertscore_f1(references, predictions, **kwargs):
    """Computes the F1 score of the BERTScore metric.

    Args:
        references: A list of reference strings.
        predictions: A list of predicted strings.
        **kwargs: Additional keyword arguments.

    Returns:
        The F1 score of the BERTScore metric.
    """
    bertscore = load("bertscore")
    return bertscore.compute(predictions=predictions, references=references, **kwargs)["f1"][0]

def bertscore_precision(references, predictions, **kwargs):
    """Computes the precision of the BERTScore metric.

    Args:
        references: A list of reference strings.
        predictions: A list of predicted strings.
        **kwargs: Additional keyword arguments.

    Returns:
        The precision of the BERTScore metric.
    """
    bertscore = load("bertscore")
    print("Predictions:", predictions)
    print("References:", references)
    return bertscore.compute(predictions=predictions, references=references, **kwargs)["precision"][0]

def bertscore_recall(references, predictions, **kwargs):
    """Computes the recall of the BERTScore metric.

    Args:
        references: A list of reference strings.
        predictions: A list of predicted strings.
        **kwargs: Additional keyword arguments.

    Returns:
        The recall of the BERTScore metric.
    """
    bertscore = load("bertscore")
    return bertscore.compute(predictions=predictions, references=references, **kwargs)["recall"][0]

def arabic_bleu(predictions, references, **kwargs):
    """
    BLEU score with Arabic-specific preprocessing
    """
    import sacrebleu
    
    # Basic Arabic text normalization (you might want to expand this)
    def normalize_arabic(text):
        # Remove diacritics, normalize spaces, etc.
        import re
        # Remove Arabic diacritics
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    # Normalize texts
    norm_predictions = [normalize_arabic(pred) for pred in predictions]
    norm_references = [[normalize_arabic(ref)] for ref in references]
    
    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(norm_predictions, list(zip(*norm_references)))
    return bleu.score / 100.0  # Convert to 0-1 scale

def arabic_chrf(predictions, references, **kwargs):
    """
    chrF score optimized for Arabic
    """
    import sacrebleu
    
    def normalize_arabic(text):
        import re
        text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
        text = ' '.join(text.split())
        return text.strip()
    
    norm_predictions = [normalize_arabic(pred) for pred in predictions]
    norm_references = [[normalize_arabic(ref)] for ref in references]
    
    chrf = sacrebleu.corpus_chrf(norm_predictions, list(zip(*norm_references)))
    return chrf.score / 100.0

