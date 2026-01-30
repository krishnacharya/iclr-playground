#!/usr/bin/env python3
"""
TF-IDF baseline for paper decision prediction.

Predicts:
- binary: accept (1) vs reject (0) using LogisticRegression
- decision: 4-class (reject, poster, spotlight, oral) using LogisticRegression
- citation: citations_normalized_by_year using LinearRegression

Text types:
- original: Full original markdown from MinerU
- clean: Title + abstract + cleaned markdown (no GitHub links, anonymized)
- clean-title: Paper title only
- original-abstract: Original abstract from OpenReview
- clean-abstract: Cleaned abstract (no GitHub links)
- clean-introduction: Introduction section extracted from content_list_json
- clean-results: Results section extracted from content_list_json
- clean-conclusion: Conclusion section extracted from content_list_json
- clean-references: References section extracted from content_list_json

Options:
--dynamically-create-clean-md: Filter line-number aside_text from content_list_json
                               and generate MD dynamically (affects clean and section types)

Usage:
    python scripts/tf_idf_analysis.py --text-type original --target binary
    python scripts/tf_idf_analysis.py --text-type clean --target citation
    python scripts/tf_idf_analysis.py --text-type clean-introduction --target binary
    python scripts/tf_idf_analysis.py --text-type clean --target binary --dynamically-create-clean-md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.prep_hf_dataset_for_analysis import (
    TEXT_TYPES,
    TARGETS,
    prepare_dataset_for_analysis,
)


def get_top_features_binary(
    vectorizer: TfidfVectorizer, model: LogisticRegression, n_features: int = 20
) -> tuple[list, list]:
    """Get top N features for accept (class 1) and reject (class 0)."""
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        # Top features for accept (highest positive coefficients)
        accept_indices = np.argsort(coefs)[-n_features:][::-1]
        accept_features = [feature_names[i] for i in accept_indices]
        # Top features for reject (lowest coefficients = most negative)
        reject_indices = np.argsort(coefs)[:n_features]
        reject_features = [feature_names[i] for i in reject_indices]
        return accept_features, reject_features
    return [], []


def get_top_features_multiclass(
    vectorizer: TfidfVectorizer, model: LogisticRegression, n_features: int = 20
) -> dict[str, list]:
    """Get top N features for each class in multiclass classification."""
    feature_names = vectorizer.get_feature_names_out()
    class_names = ["reject", "poster", "spotlight", "oral"]
    result = {}
    if hasattr(model, "coef_"):
        for i, class_name in enumerate(class_names):
            if i < model.coef_.shape[0]:
                coefs = model.coef_[i]
                top_indices = np.argsort(coefs)[-n_features:][::-1]
                result[class_name] = [feature_names[j] for j in top_indices]
    return result


def get_top_features_by_quartile(
    vectorizer: TfidfVectorizer, X, y_normalized: np.ndarray, n_features: int = 15
) -> dict[str, list]:
    """Get top N TF-IDF features for each citation quartile."""
    quartiles = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
    bounds = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    feature_names = vectorizer.get_feature_names_out()
    result = {}
    for q_name, (low, high) in zip(quartiles, bounds):
        if high < 1:
            mask = (y_normalized >= low) & (y_normalized < high)
        else:
            mask = (y_normalized >= low) & (y_normalized <= high)
        if mask.sum() > 0:
            X_subset = X[mask]
            # Mean TF-IDF score per feature in this quartile
            mean_tfidf = np.asarray(X_subset.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_tfidf)[-n_features:][::-1]
            result[q_name] = [feature_names[i] for i in top_indices]
    return result


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline for paper decision prediction")
    parser.add_argument(
        "--text-type",
        choices=TEXT_TYPES,
        required=True,
        help="Text source: original/clean MD, title, abstract, or section"
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        required=True,
        help="Prediction target",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/iclr_2020_2025_80_20_split",
        help="Path to HF dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/generated_tf_idf_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dynamically-create-clean-md",
        action="store_true",
        help="Filter content_list_json to remove line-number asides before text extraction"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.text_type / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TF-IDF ANALYSIS")
    print("=" * 60)
    print(f"Text type: {args.text_type}")
    print(f"Target: {args.target}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load and prepare dataset using shared library
    print("\n1. Preparing dataset...")
    train_texts_f, train_y, test_texts_f, test_y = prepare_dataset_for_analysis(
        dataset_path=args.dataset_path,
        text_type=args.text_type,
        target=args.target,
        filter_asides=args.dynamically_create_clean_md,
        verbose=True
    )

    print(f"\n2. Final samples:")
    print(f"   Train samples: {len(train_y):,}")
    print(f"   Test samples: {len(test_y):,}")

    # TF-IDF
    print("\n3. Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000, min_df=5, max_df=0.95, ngram_range=(1, 2)
    )
    X_train = vectorizer.fit_transform(train_texts_f)
    X_test = vectorizer.transform(test_texts_f)
    print(f"   Feature shape: {X_train.shape}")

    # Model training and evaluation
    print("\n4. Training model...")
    if args.target == "citation":
        model = LinearRegression()
        model.fit(X_train, train_y)
        predictions = model.predict(X_test)

        # Correlation coefficient
        corr, p_value = pearsonr(test_y, predictions)
        results = {
            "text_type": args.text_type,
            "target": args.target,
            "correlation": corr,
            "p_value": p_value,
            "train_size": len(train_y),
            "test_size": len(test_y),
        }
        print(f"   Correlation: {corr:.4f} (p={p_value:.2e})")
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, train_y)
        predictions = model.predict(X_test)

        # For decision (4-class), compute binary metrics by mapping poster/spotlight/oral -> 1
        if args.target == "decision":
            # Map to binary for metrics: reject=0, others=1
            test_y_binary = (test_y > 0).astype(int)
            pred_binary = (predictions > 0).astype(int)
        else:
            test_y_binary = test_y
            pred_binary = predictions

        results = {
            "text_type": args.text_type,
            "target": args.target,
            "accuracy": accuracy_score(test_y_binary, pred_binary),
            "precision": precision_score(test_y_binary, pred_binary, average="binary"),
            "recall": recall_score(test_y_binary, pred_binary, average="binary"),
            "f1": f1_score(test_y_binary, pred_binary, average="binary"),
            "train_size": len(train_y),
            "test_size": len(test_y),
        }
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   F1: {results['f1']:.4f}")

    # Save results CSV
    print("\n5. Saving results...")
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / "test_results.csv", index=False)
    print(f"   Results saved to {output_dir / 'test_results.csv'}")

    # Generate features CSV based on target type
    print("\n6. Extracting top features...")
    if args.target == "binary":
        # Binary: 2 columns (accept, reject) with 20 features each
        accept_features, reject_features = get_top_features_binary(
            vectorizer, model, n_features=20
        )
        features_df = pd.DataFrame({"accept": accept_features, "reject": reject_features})
        print(f"   Top accept features: {accept_features[:5]}")
        print(f"   Top reject features: {reject_features[:5]}")
    elif args.target == "decision":
        # Decision: 4 columns (reject, poster, spotlight, oral) with 20 features each
        features_dict = get_top_features_multiclass(vectorizer, model, n_features=20)
        # Pad to equal length if needed
        max_len = max(len(v) for v in features_dict.values()) if features_dict else 0
        for k in features_dict:
            features_dict[k] += [""] * (max_len - len(features_dict[k]))
        features_df = pd.DataFrame(features_dict)
        for cls, feats in features_dict.items():
            print(f"   Top {cls} features: {feats[:5]}")
    elif args.target == "citation":
        # Citation: 4 columns for quartiles with 15 features each
        features_dict = get_top_features_by_quartile(
            vectorizer, X_test, test_y, n_features=15
        )
        # Pad to equal length
        max_len = max(len(v) for v in features_dict.values()) if features_dict else 0
        for k in features_dict:
            features_dict[k] += [""] * (max_len - len(features_dict[k]))
        features_df = pd.DataFrame(features_dict)
        for q, feats in features_dict.items():
            print(f"   Quartile {q}: {feats[:5]}")

    features_df.to_csv(output_dir / "generated_features.csv", index=False)
    print(f"   Features saved to {output_dir / 'generated_features.csv'}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
