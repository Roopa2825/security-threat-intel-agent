"""Use the fine-tuned phishing classifier (train it first: python -m app.train_phishing_model)."""
import os
from functools import lru_cache

from app.train_phishing_model import MAX_CHARS, MAX_TOKENS, MODEL_DIR


def verdict_for(probability: float) -> str:
    if probability >= 0.8:
        return "likely phishing"
    if probability >= 0.5:
        return "possibly phishing"
    return "likely legitimate"


@lru_cache(maxsize=1)
def get_classifier():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        raise FileNotFoundError(
            f"No trained model in {MODEL_DIR}. Run: python -m app.train_phishing_model"
        )
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR,
                    truncation=True, max_length=MAX_TOKENS, top_k=None)


def classify_text(text: str) -> dict:
    """Return the probability that the text is phishing, with a verdict."""
    scores = get_classifier()(text[:MAX_CHARS])
    if scores and isinstance(scores[0], list):  # some versions nest the result
        scores = scores[0]
    probability = next(s["score"] for s in scores if s["label"] == "phishing")
    return {
        "phishing_probability": round(float(probability), 3),
        "verdict": verdict_for(probability),
        "model": "DistilBERT fine-tuned on a public phishing email dataset",
        "note": "A statistical model: treat this as one signal, not proof.",
    }
