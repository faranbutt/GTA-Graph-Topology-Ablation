import os
import sys
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score



def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scoring_script.py path/to/submission.csv")

    submission_path = sys.argv[1]

    data_dir = os.path.join("gnn_challenge", "data")
    truth_path = os.path.join(data_dir, "test_labels_hidden.csv")

    sub = pd.read_csv(submission_path)
    truth = pd.read_csv(truth_path)

    # Basic checks
    required_sub_cols = {"filename", "prediction"}
    required_truth_cols = {"filename", "target"}

    if not required_sub_cols.issubset(sub.columns):
        raise ValueError(f"Submission must have columns {sorted(required_sub_cols)}")
    if not required_truth_cols.issubset(truth.columns):
        raise ValueError(f"Hidden labels must have columns {sorted(required_truth_cols)}")

    # Merge by filename so row order doesn't matter
    merged = sub.merge(truth, on="filename", how="inner")

    if len(merged) != len(truth):
        raise ValueError(
            f"Filename mismatch. Matched {len(merged)} rows, but truth has {len(truth)} rows."
        )

    y_true = merged["target"].astype(int)
    y_pred = merged["prediction"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")

if __name__ == "__main__":
    main()
