"""Week 3: fine-tune a Hugging Face Transformer (DistilBERT) to classify email text
as phishing or safe, and compare it with a classic baseline (TF-IDF + logistic regression).

Data: zefang-liu/phishing-email-dataset on Hugging Face (copy of the Kaggle
"Phishing Email Detection" dataset, LGPL-3.0).

Run:  python -m app.train_phishing_model                 (about 20-40 minutes on a laptop CPU)
      python -m app.train_phishing_model --train 1000    (quick test run)
"""
import argparse
import json
import os
import random

from app import config  # noqa: F401  (loads .env and quiet settings first)

DATASET = "zefang-liu/phishing-email-dataset"
BASE_MODEL = "distilbert-base-uncased"
MODEL_DIR = os.getenv("PHISHING_MODEL_DIR", "models/phishing-distilbert")
LABELS = {"Safe Email": 0, "Phishing Email": 1}
MAX_TOKENS = 256
MAX_CHARS = 5000


def clean_rows(rows: list[dict]) -> list[dict]:
    """Keep labeled, non-trivial emails and drop exact duplicates. Duplicates matter:
    the same email in both train and test would inflate the scores (data leakage)."""
    seen, cleaned = set(), []
    for row in rows:
        text = (row.get("Email Text") or "").strip()
        label = LABELS.get(row.get("Email Type"))
        if label is None or len(text) < 20 or text.lower() == "empty":
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"text": text[:MAX_CHARS], "label": label})
    return cleaned


def split(rows: list[dict], n_train: int, n_val: int, n_test: int, seed: int) -> tuple[list, list, list]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    test = rows[:n_test]
    val = rows[n_test:n_test + n_val]
    train = rows[n_test + n_val:n_test + n_val + n_train]
    return train, val, test


def classification_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    predictions = [int(p >= threshold) for p in probabilities]
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision": round(precision_score(labels, predictions, zero_division=0), 4),
        "recall": round(recall_score(labels, predictions, zero_division=0), 4),
        "f1": round(f1_score(labels, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probabilities), 4),
    }


def best_threshold(labels: list[int], probabilities: list[float]) -> float:
    """Pick the cutoff that maximizes F1 on the validation set. Each model gets its own
    cutoff, so the comparison is fair (a fixed 0.5 can be wrong for one of them)."""
    from sklearn.metrics import f1_score

    candidates = [round(0.05 * i, 2) for i in range(1, 20)]
    return max(candidates, key=lambda t: f1_score(labels, [int(p >= t) for p in probabilities], zero_division=0))


def evaluate_both_ways(val_labels, val_probs, test_labels, test_probs) -> dict:
    """Test metrics at the default 0.5 cutoff and at the cutoff tuned on validation data."""
    threshold = best_threshold(val_labels, val_probs)
    return {
        "threshold": threshold,
        "at_0.5": classification_metrics(test_labels, test_probs, 0.5),
        "tuned": classification_metrics(test_labels, test_probs, threshold),
    }


def train_baseline(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    """A strong, fast classic baseline. A Transformer is only worth it if it beats this."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    model = LogisticRegression(max_iter=1000)
    model.fit(vectorizer.fit_transform([r["text"] for r in train]), [r["label"] for r in train])

    def probabilities(rows):
        return list(model.predict_proba(vectorizer.transform([r["text"] for r in rows]))[:, 1])

    return evaluate_both_ways([r["label"] for r in val], probabilities(val),
                              [r["label"] for r in test], probabilities(test))


def train_transformer(train: list[dict], val: list[dict], test: list[dict], epochs: float, seed: int) -> dict:
    import numpy as np
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "safe", 1: "phishing"}, label2id={"safe": 0, "phishing": 1}
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_TOKENS)

    datasets = {name: Dataset.from_list(rows).map(tokenize, batched=True)
                for name, rows in {"train": train, "val": val, "test": test}.items()}

    args = TrainingArguments(
        output_dir="models/_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"],
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()

    def probabilities(split_name: str) -> list[float]:
        logits = trainer.predict(datasets[split_name]).predictions
        return list((np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[:, 1])

    results = evaluate_both_ways(list(datasets["val"]["label"]), probabilities("val"),
                                 list(datasets["test"]["label"]), probabilities("test"))
    os.makedirs(MODEL_DIR, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    return results


def main() -> None:
    import mlflow
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Fine-tune a phishing email classifier.")
    parser.add_argument("--train", type=int, default=3000, help="training emails (more = slower, usually better)")
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Downloading {DATASET}...")
    rows = clean_rows(list(load_dataset(DATASET, split="train")))
    print(f"{len(rows)} unique labeled emails ({sum(r['label'] for r in rows)} phishing)")
    train, val, test = split(rows, args.train, args.val, args.test, args.seed)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("phishing-classifier")
    with mlflow.start_run(run_name=f"{BASE_MODEL}-vs-tfidf"):
        mlflow.log_params({"dataset": DATASET, "base_model": BASE_MODEL, "train_size": len(train),
                           "test_size": len(test), "epochs": args.epochs, "max_tokens": MAX_TOKENS,
                           "seed": args.seed})

        print("\nTraining baseline (TF-IDF + logistic regression)...")
        baseline = train_baseline(train, val, test)
        print("  baseline test (tuned cutoff):", baseline["tuned"])

        print("\nFine-tuning DistilBERT (this is the slow part)...")
        distilbert = train_transformer(train, val, test, args.epochs, args.seed)
        print("  distilbert test (tuned cutoff):", distilbert["tuned"])

        for name, result in {"baseline": baseline, "distilbert": distilbert}.items():
            mlflow.log_metric(f"{name}_threshold", result["threshold"])
            mlflow.log_metrics({f"{name}_{k}": v for k, v in result["tuned"].items()})
            mlflow.log_metrics({f"{name}_at05_{k}": v for k, v in result["at_0.5"].items()})

        summary = {"baseline": baseline, "distilbert": distilbert, "train_size": len(train)}
        with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_dict(summary, "phishing_classifier_metrics.json")

    print(f"\nSaved model to {MODEL_DIR}")
    print(f"\nTest results with each model's cutoff tuned on validation data "
          f"(TF-IDF {baseline['threshold']}, DistilBERT {distilbert['threshold']}):")
    print(f"{'Metric':<10}{'TF-IDF':>10}{'DistilBERT':>12}")
    for key in baseline["tuned"]:
        print(f"{key:<10}{baseline['tuned'][key]:>10}{distilbert['tuned'][key]:>12}")

if __name__ == "__main__":
    main()
