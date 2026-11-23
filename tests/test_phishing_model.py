from app.phishing_model import verdict_for
from app.train_phishing_model import clean_rows, split


def test_clean_rows_drops_duplicates_unlabeled_and_tiny_texts():
    rows = [
        {"Email Text": "Please verify your account password now at this link.", "Email Type": "Phishing Email"},
        {"Email Text": "please   verify your account password now at this link.", "Email Type": "Phishing Email"},
        {"Email Text": "empty", "Email Type": "Safe Email"},
        {"Email Text": "Short", "Email Type": "Safe Email"},
        {"Email Text": "Meeting notes from Tuesday are attached for review.", "Email Type": "Unknown"},
        {"Email Text": "Meeting notes from Tuesday are attached for review.", "Email Type": "Safe Email"},
    ]
    cleaned = clean_rows(rows)
    assert [r["label"] for r in cleaned] == [1, 0]


def test_split_has_no_overlap():
    rows = [{"text": f"email number {i}", "label": i % 2} for i in range(100)]
    train, val, test = split(rows, 60, 10, 20, seed=1)
    texts = [set(r["text"] for r in part) for part in (train, val, test)]
    assert len(train) == 60 and len(val) == 10 and len(test) == 20
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])


def test_verdicts():
    assert verdict_for(0.95) == "likely phishing"
    assert verdict_for(0.6) == "possibly phishing"
    assert verdict_for(0.1) == "likely legitimate"
