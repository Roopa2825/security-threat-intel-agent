"""Use the fine-tuned phishing classifier (train it first: python -m app.train_phishing_model)."""
import json
import os
from functools import lru_cache

from app.train_phishing_model import MAX_CHARS, MAX_TOKENS, MODEL_DIR


def verdict_for(probability: float, threshold: float = 0.5) -> str:
    """'possibly phishing' starts at the cutoff tuned on validation data during training."""
    if probability >= max(0.8, threshold):
        return "likely phishing"
    if probability >= threshold:
        return "possibly phishing"
    return "likely legitimate"


@lru_cache(maxsize=1)
def tuned_threshold() -> float:
    """The cutoff chosen on validation data during training (saved in metrics.json)."""
    try:
        with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
            return float(json.load(f)["distilbert"]["threshold"])
    except (OSError, KeyError, ValueError, TypeError):
        return 0.5


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
        "verdict": verdict_for(probability, tuned_threshold()),
        "threshold": tuned_threshold(),
        "model": "DistilBERT fine-tuned on a public phishing email dataset",
        "note": "A statistical model: treat this as one signal, not proof.",
    }
