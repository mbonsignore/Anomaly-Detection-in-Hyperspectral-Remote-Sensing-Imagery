import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)


def load_scores_from_single_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads scores and labels from a single CSV containing:
        image_name, anomaly_score, label

    Returns:
        scores (np.ndarray): anomaly scores
        labels (np.ndarray): 0 = normal, 1 = anomalous
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "score" in df.columns:
        print("[WARNING] 'score' column found; renaming to 'anomaly_score' for consistency.")
        df = df.rename(columns={"score": "anomaly_score"})

    required_cols = {"anomaly_score", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, but got: {df.columns.tolist()}"
        )

    scores = df["anomaly_score"].values.astype(float)
    labels = df["label"].values.astype(int)

    return scores, labels


def evaluate(scores, labels, strategy="f1", percentile=95):
    """Same as before: computes threshold, preds, and metrics"""

    if strategy == "f1":
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresholds = precision_recall_curve(labels, scores)

        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        threshold = thresholds[best_idx]

    elif strategy == "percentile":
        threshold = np.percentile(scores[labels == 0], percentile)
        #Floga inverted configuration
        #threshold = np.percentile(scores[labels == 1], percentile)
    else:
        raise ValueError("Invalid threshold strategy.")

    preds = (scores > threshold).astype(int)
    #Floga inverted configuration
    #preds = (scores < threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": (preds == labels).mean(),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1_score": f1_score(labels, preds),
        "auroc": roc_auc_score(labels, scores),
        #"auroc": roc_auc_score(labels, -scores),  # Floga inverted configuration
        "confusion_matrix": confusion_matrix(labels, preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="../../../data/datasets/PatchCore/results/copernicusfm",
                        help="Top-level results folder.")
    parser.add_argument("--dataset", type=str,
                        help="Dataset name (e.g., 'hyperseg').")
    parser.add_argument("--sampling", type=str, choices=["greedy", "random"])
    parser.add_argument("--strategy", type=str, choices=["f1", "percentile"], default="f1")
    parser.add_argument("--sampling-ratio", type=float, default=0.1, help="Sampling ratio used.")
    parser.add_argument("--percentile", type=int, default=95,
                        help="Percentile if strategy is 'percentile'")
    parser.add_argument("--mixed", type=bool, default=False, help="Whether to use mixed normal features.")
    args = parser.parse_args()

    # expected path: results/<dataset>/<sampling>/scores.csv
    if args.mixed:
        csv_path = Path("../../../data/datasets/PatchCore/results/copernicusfm/mixed/spectralearth7bandsbankcopernicuspretrainnormalbalanced.csv")   
    else:
        csv_path = Path(args.results_root) / args.dataset / args.sampling / f"{args.sampling_ratio}.csv"
    print(f"[INFO] Reading scores from: {csv_path}")

    scores, labels = load_scores_from_single_csv(csv_path)

    print(f"[INFO] Loaded {len(scores)} samples "
          f"({(labels==1).sum()} anomalous, {(labels==0).sum()} normal)")

    metrics = evaluate(scores, labels, strategy=args.strategy, percentile=args.percentile)

    print("\n==== Evaluation Metrics ====")
    for key, val in metrics.items():
        if key == "confusion_matrix":
            print("Confusion Matrix:\n", val)
        else:
            print(f"{key:>16}: {val:.4f}")


if __name__ == "__main__":
    main()

'''
python3 image_level_metrics.py --dataset flogatestfinetuningsemisupervisedfourthrun --sampling random --strategy f1 --sampling-ratio 0.1
python3 image_level_metrics.py --dataset flogatestfinetuningsemisupervised --sampling random --strategy percentile --sampling-ratio 0.1
python3 image_level_metrics.py --mixed True
python3 image_level_metrics.py --dataset floganormal --sampling random --strategy f1 --sampling-ratio 0.1
'''