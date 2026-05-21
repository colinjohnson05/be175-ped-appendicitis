import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_pp import appendicitis_pp


DEFAULT_DATA_PATHS = [
    "data/app_data.xlsx",
    "app_data.xlsx",
    "app_data (3).xlsx",
]

NON_FEATURE_COLUMNS = [
    "Diagnosis",
    "Class",
    "Diagnosis_Presumptive",
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

def train_xgboost_baseline(data_path, test_size=0.2, random_state=42):
    data = appendicitis_pp(data_path)

    feature_columns = [
        column for column in data.columns if column not in NON_FEATURE_COLUMNS
    ]
    X = data[feature_columns]
    y = data["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
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
        "feature_importance": pd.Series(
            model.feature_importances_,
            index=feature_columns,
        ).sort_values(ascending=False),
    }

    return model, metrics

def train_xgboost(X_train, X_test, y_train, y_test, random_state:int = 42):
    """
Trains XGBoost model on ``data`` using cross validation method ``cv``.
    :param X_train: Pandas dataframe containing data being used to train XGBoost model.
    :param X_test: Pandas dataframe containing test or validation dataset
    :param y_train: Pandas dataframe containing labels for training data.
    :param y_test: Pandas dataframe containing labels for testing or validation data.
    :param random_state: Random state to ensure reproducibility
    :return: Model and metrics
    """

    # Calculate positive:negative ratio for gradient scaling in xgboost
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    feature_columns = X_train.columns.tolist()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
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
        "feature_importance": pd.Series(
            model.feature_importances_,
            index=feature_columns,
        ).sort_values(ascending=False),
    }

    return model, metrics, y_prob


def print_metrics(metrics, top_n_features=15):
    print("\nXGBoost Baseline")
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

    print(f"\nTop {top_n_features} feature importances:")
    print(metrics["feature_importance"].head(top_n_features))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost baseline for appendicitis prediction."
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
    data = appendicitis_pp(resolve_data_path(args.data))
    # split 0.8 train 0.2 test just to make sure code runs
    X = data.drop(columns=['Class', 'Diagnosis', 'Diagnosis_Presumptive'])
    y = data['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    model, baseline_metrics, y_prob = train_xgboost(X_train, X_test, y_train, y_test, random_state=args.random_state)
    print_metrics(baseline_metrics)
    print(y_prob)
