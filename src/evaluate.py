"""
Evaluate SpendingClassifier accuracy using 5-fold stratified cross-validation.

Embeddings are computed once up-front (using the shared cache) so each fold
only incurs the cost of CatBoost training and inference — not re-embedding.

Usage:
    python src/evaluate.py
    python src/evaluate.py --master master.csv --cache embedding_cache.db --folds 5
    python src/evaluate.py --plot          # also save a confusion-matrix heatmap
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import EmbeddingCache, get_embeddings

MIN_SAMPLES_PER_CATEGORY = 5  # categories with fewer samples are excluded


def load_and_prepare(master_csv, cache_db):
    df = pd.read_csv(master_csv)
    df = df.dropna(subset=["category", "description"])
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0)

    # Drop categories too rare for stratified splitting
    counts = df["category"].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_CATEGORY].index.tolist()
    if rare:
        print(f"Excluding {len(rare)} rare categor{'y' if len(rare)==1 else 'ies'} "
              f"(< {MIN_SAMPLES_PER_CATEGORY} samples): {rare}")
        df = df[~df["category"].isin(rare)]

    print(f"Dataset: {len(df)} transactions, {df['category'].nunique()} categories")

    print("Computing embeddings (cached where possible)...")
    cache = EmbeddingCache(cache_db)
    embeddings = np.array(get_embeddings(df["description"].tolist(), cache=cache))
    other = df[["debit", "credit"]].values
    X = np.hstack([embeddings, other])
    y = df["category"].values

    return X, y


def run_cv(X, y, n_folds, catboost_kwargs):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = CatBoostClassifier(**catboost_kwargs, verbose=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test).flatten()

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_true.extend(y_test)
        all_pred.extend(y_pred)

        print(f"  Fold {fold}/{n_folds}: accuracy = {acc:.4f}")

    return fold_accuracies, np.array(all_true), np.array(all_pred)


def print_results(fold_accuracies, all_true, all_pred):
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nMean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    print("\nClassification Report (aggregated across all folds)")
    print("=" * 60)
    print(classification_report(all_true, all_pred, digits=3, zero_division=0))


def plot_confusion_matrix(all_true, all_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(all_true))
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix (5-fold CV)")
    plt.tight_layout()
    out_path = "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpendingClassifier via k-fold CV.")
    parser.add_argument("--master", default="master.csv", help="Path to master CSV")
    parser.add_argument("--cache", default="embedding_cache.db", help="Path to embedding cache DB")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--plot", action="store_true", help="Save confusion matrix heatmap")
    args = parser.parse_args()

    catboost_kwargs = dict(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
    )

    X, y = load_and_prepare(args.master, args.cache)

    print(f"\n{args.folds}-Fold Cross-Validation")
    print("=" * 30)
    fold_accuracies, all_true, all_pred = run_cv(X, y, args.folds, catboost_kwargs)

    print_results(fold_accuracies, all_true, all_pred)

    if args.plot:
        plot_confusion_matrix(all_true, all_pred)


if __name__ == "__main__":
    main()
