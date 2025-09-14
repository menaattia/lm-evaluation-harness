from evaluate import load
import re
import unicodedata
from openai import OpenAI
from lm_eval.api.metrics import register_metric

# @register_metric(
#     metric="arabic_exact_match",
#     higher_is_better=True,
#     output_type="generate_until",
#     aggregation="mean",
# )
# def arabic_exact_match(predictions, references, **kwargs):
#     """
#     Exact match for Arabic text with normalization
#     """
#     def normalize_arabic(text):
#         # Remove diacritics and normalize
#         text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)  # Remove diacritics

#         # Normalize yeh variations (Persian/Urdu yeh to Arabic yeh)
#         text = text.replace('ی', 'ي')  # Persian/Urdu yeh to Arabic yeh
#         text = text.replace('ى', 'ي')  # Alef maksura to Arabic yeh
        
#         # Normalize alef variations
#         text = text.replace('أ', 'ا')  # Alef with hamza above
#         text = text.replace('إ', 'ا')  # Alef with hamza below
#         text = text.replace('آ', 'ا')  # Alef with madda

#         # Remove common punctuation
#         text = re.sub(r'[.,;:!?؟،؛]', '', text)

#         text = text.strip()
#         return text
    
#     scores = []
#     for pred, ref in zip(predictions, references):
#         pred_norm = normalize_arabic(pred)
#         ref_norm = normalize_arabic(ref)
#         scores.append(1.0 if pred_norm == ref_norm else 0.0)
    
#     return sum(scores) / len(scores)
def arabic_exact_match(predictions, references, **kwargs):
    """
    Exact match for Arabic text with comprehensive normalization
    """
    def normalize_arabic(text):
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', str(text))
        
        # Remove diacritics (extended range)
        text = re.sub(r'[\u064B-\u065F\u0670\u0640\u06D6-\u06ED]', '', text)
        
        # Normalize alef variations
        text = text.replace('أ', 'ا')  # Alef with hamza above
        text = text.replace('إ', 'ا')  # Alef with hamza below
        text = text.replace('آ', 'ا')  # Alef with madda
        text = text.replace('ٱ', 'ا')  # Alef wasla
        
        # Normalize yeh variations
        text = text.replace('ی', 'ي')  # Persian/Urdu yeh
        text = text.replace('ى', 'ي')  # Alef maksura
        text = text.replace('ئ', 'ي')  # Yeh with hamza
        
        # Normalize heh variations
        text = text.replace('ة', 'ه')  # Teh marbuta
        text = text.replace('ۃ', 'ه')  # Teh marbuta goal (Urdu)
        
        # Remove punctuation
        text = re.sub(r'[.,;:!?؟،؛٪٫٬؍﴾﴿﷼﷽"\'\-_(){}[\]<>]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_arabic(pred)
        ref_norm = normalize_arabic(ref)
        scores.append(1.0 if pred_norm == ref_norm else 0.0)
    
    return sum(scores) / len(scores) if scores else 0.0

def english_exact_match(predictions, references, **kwargs):
    """
    Exact match for English text with normalization
    """
    def normalize_english(text):
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', str(text))
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common punctuation
        text = re.sub(r'[.,;:!?"\'\-_(){}[\]<>/\\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_english(pred)
        ref_norm = normalize_english(ref)
        scores.append(1.0 if pred_norm == ref_norm else 0.0)
    
    return sum(scores) / len(scores) if scores else 0.0

    
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
    kwargs['device'] = 'cpu'
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


def llm_judge_score(predictions=None, references=None, **kwargs):
    client = OpenAI()
    if predictions is None or references is None:
        raise ValueError("Missing predictions or references")

    scores = []

    for pred, ref in zip(predictions, references):
        gold_explanation = ref
        generated_explanation = pred

        prompt = f"""You are an expert in Arabic language and culture. Your task is to evaluate how well a generated explanation matches the intended meaning of the reference explanation of an Arabic idiom/proverb.
                    Below is the reference (gold) explanation and a generated explanation.

                    Rate the accuracy of the generated explanation based on how well it preserves the intended meaning of the idiom/proverb.

                    Gold Explanation: {gold_explanation}
                    Generated Explanation: {generated_explanation}

                    Use the following rating scale:

                    5 = Excellent — Perfectly matches the gold explanation in meaning.
                    4 = Good — Minor omissions or phrasing differences, but the meaning is well preserved.
                    3 = Fair — Partial understanding, some inaccuracies or missing key aspects.
                    2 = Poor — Significant misunderstanding or loss of core meaning.
                    1 = Very Poor — Completely incorrect or irrelevant explanation.

                    Only output a numerical rating and nothing else.
                    Rating (1–5):"""

        response = client.responses.create(
            model="gpt-4.1-2025-04-14",
            input=prompt,
            temperature=0
        )

        reply = response.output_text.strip()

        # Extract numeric rating
        score = next((int(token) for token in reply.split() if token.isdigit() and 1 <= int(token) <= 5), None)
        if score is None:
            raise ValueError(f"Unexpected model response: {reply}")

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


import anthropic

def llm_judge_score_claude(predictions=None, references=None, **kwargs):
    client = anthropic.Anthropic()
    if predictions is None or references is None:
        raise ValueError("Missing predictions or references")

    scores = []

    for pred, ref in zip(predictions, references):
        gold_explanation = ref
        generated_explanation = pred

        prompt = f"""You are an expert in Arabic language and culture. Your task is to evaluate how well a generated explanation matches the intended meaning of the reference explanation of an Arabic idiom/proverb.
            Below is the reference (gold) explanation and a generated explanation.

            Rate the accuracy of the generated explanation based on how well it preserves the intended meaning of the idiom/proverb.

            Gold Explanation: {gold_explanation}
            Generated Explanation: {generated_explanation}

            Use the following rating scale:

            5 = Excellent — Perfectly matches the gold explanation in meaning.
            4 = Good — Minor omissions or phrasing differences, but the meaning is well preserved.
            3 = Fair — Partial understanding, some inaccuracies or missing key aspects.
            2 = Poor — Significant misunderstanding or loss of core meaning.
            1 = Very Poor — Completely incorrect or irrelevant explanation.

            Only output a numerical rating and nothing else.
            Rating (1–5):"""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        reply = response.content[0].text.strip()

        # Extract numeric rating
        score = next((int(token) for token in reply.split() if token.isdigit() and 1 <= int(token) <= 5), None)
        if score is None:
            raise ValueError(f"Unexpected model response: {reply}")

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0



from collections import defaultdict

def clean_sentiment(s):
    """
    Clean up model outputs for more robust matching.
    Returns lowercase and only the first word if there are extras.
    """
    return s.strip().lower().split()[0]

def sentiment_match(predictions, docs, **kwargs):
    """
    Assumes that predictions and docs are in the order:
    [proverb_0, explanation_0, proverb_1, explanation_1, ...]
    Computes % where sentiment(proverb) == sentiment(explanation)
    """
    # Group predictions by id
    by_id = defaultdict(dict)
    for pred, doc in zip(predictions, docs):
        by_id[doc['id']][doc['phase']] = clean_sentiment(pred)
    
    n = 0
    match = 0
    mismatches = []
    for id_, sent_dict in by_id.items():
        if 'proverb' in sent_dict and 'explanation' in sent_dict:
            n += 1
            if sent_dict['proverb'] == sent_dict['explanation']:
                match += 1
            else:
                mismatches.append((id_, sent_dict['proverb'], sent_dict['explanation']))
        # else skip incomplete pairs
    
    score = match / n if n > 0 else 0.0
    # Optionally: return mismatches for analysis
    return {
        "sentiment_match": score,
        "n_samples": n,
        "mismatches": mismatches[:20]  # first 20 only for debugging
    }