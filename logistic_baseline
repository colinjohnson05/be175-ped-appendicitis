import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pp import appendicitis_pp


DEFAULT_DATA_PATHS = [
    "data/app_data.xlsx",
    "app_data.xlsx",
    "app_data (3).xlsx",
]


def resolve_data_path(data_path):
    if data_path:
        return data_path

    for candidate in DEFAULT_DATA_PATHS:
        if Path(candidate).exists():
            return candidate

    searched = ", ".join(DEFAULT_DATA_PATHS)
    raise FileNotFoundError(
        f"No dataset found. Expected one of: {searched}. "
        "You can also pass a path with --data."
    )


def specificity_score(y_true, y_pred):
    """Return true negative rate for binary labels where appendicitis is 1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def train_logistic_baseline(data_path, test_size=0.2, random_state=42):
    data = appendicitis_pp(data_path)

    X = data.drop(columns=["Diagnosis"])
    y = data["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "n_total": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "specificity": specificity_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["No appendicitis", "Appendicitis"],
        ),
    }

    return model, metrics


def print_metrics(metrics):
    print("\nLogistic Regression Baseline")
    print(f"Total samples: {metrics['n_total']}")
    print(f"Train samples: {metrics['n_train']}")
    print(f"Test samples: {metrics['n_test']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual no appendicitis", "Actual appendicitis"],
        columns=["Predicted no appendicitis", "Predicted appendicitis"],
    )
    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(metrics["classification_report"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a logistic regression baseline for appendicitis prediction."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the raw appendicitis Excel dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, baseline_metrics = train_logistic_baseline(
        data_path=resolve_data_path(args.data),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print_metrics(baseline_metrics)
