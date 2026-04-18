"""
Data loader for COMP9417 Project.
Downloads, cleans, and splits 5 tabular datasets into 60/20/20 Train/Val/Test sets.
All splits use random_state=42 for reproducibility.
Classification tasks use stratified splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _split_data(X, y, task="classification", seed=RANDOM_STATE):
    """
    Split data into 60% train, 20% val, 20% test.
    Classification tasks use stratified splits.
    """
    stratify = y if task == "classification" else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )

    stratify_train = y_trainval if task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=seed,
        stratify=stratify_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _save_split(X, y, split_name, dataset_name):
    """Save a single split to data/<dataset_name>/<split_name>."""
    split_dir = os.path.join(DATA_DIR, dataset_name, split_name)
    _ensure_dir(split_dir)
    X.to_csv(os.path.join(split_dir, "X.csv"), index=False)
    if isinstance(y, pd.DataFrame):
        y.to_csv(os.path.join(split_dir, "y.csv"), index=False)
    else:
        y.to_frame().to_csv(os.path.join(split_dir, "y.csv"), index=False)


def load_dry_bean():
    """
    Dry Bean Dataset (UCI ID: 602).
    Multi-class classification, purely numerical features.
    """
    print("Loading Dry Bean dataset...")
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    X = X.copy()
    y = y.copy()

    print(f"  Shape: {X.shape}, Classes: {y.nunique()}, Target: {y.name}")

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, task="classification"
    )

    _save_split(X_train, y_train, "train", "dry_bean")
    _save_split(X_val, y_val, "val", "dry_bean")
    _save_split(X_test, y_test, "test", "dry_bean")

    print("  Dry Bean dataset saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_superconductivity():
    """
    Superconductivity Data (UCI ID: 464).
    Regression task, massive sample size (n = 21263).
    Purely IID physical data, NO time-series leakage risk.
    Perfect for exposing SVM's O(n^2) scaling limits!
    """
    print("Loading Superconductivity dataset...")
    dataset = fetch_ucirepo(id=464)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    X = X.copy()
    y = y.copy()

    print(f"  Shape: {X.shape}, Target: {y.name}")

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, task="regression"
    )

    _save_split(X_train, y_train, "train", "superconductivity")
    _save_split(X_val, y_val, "val", "superconductivity")
    _save_split(X_test, y_test, "test", "superconductivity")

    print("  Superconductivity dataset saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_online_news():
    """
    Online News Popularity (UCI ID: 332).
    Regression task, high-dimensional (d > 50).
    Drops 'url' and 'timedelta' columns as they are non-predictive identifiers.
    """
    print("Loading Online News dataset...")
    dataset = fetch_ucirepo(id=332)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    X = X.copy()
    y = y.copy()

    drop_cols = ["url", "timedelta"]
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])
            print(f"  Dropped column: {col}")

    print(f"  Shape after cleaning: {X.shape}, Target: {y.name}")

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, task="regression"
    )

    _save_split(X_train, y_train, "train", "online_news")
    _save_split(X_val, y_val, "val", "online_news")
    _save_split(X_test, y_test, "test", "online_news")

    print("  Online News dataset saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_ai4i_maintenance():
    """
    AI4I 2020 Predictive Maintenance Dataset (UCI ID: 601).
    Binary classification, mixed numerical and categorical features.
    Drops 'UID' and 'Product ID' as they are unique identifiers.
    """
    print("Loading AI4I Predictive Maintenance dataset...")
    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    X = X.copy()
    y = y.copy()

    drop_cols = ["UID", "Product ID"]
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])
            print(f"  Dropped column: {col}")

    print(f"  Shape after cleaning: {X.shape}, Target distribution:\n{y.value_counts()}")

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, task="classification"
    )

    _save_split(X_train, y_train, "train", "ai4i_maintenance")
    _save_split(X_val, y_val, "val", "ai4i_maintenance")
    _save_split(X_test, y_test, "test", "ai4i_maintenance")

    print("  AI4I dataset saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_bank_marketing():
    """
    Bank Marketing Dataset (UCI ID: 222).
    Binary classification, imbalanced social science data.
    Maps target 'y' to {'yes': 1, 'no': 0}.
    """
    print("Loading Bank Marketing dataset...")
    dataset = fetch_ucirepo(id=222)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    X = X.copy()
    y = y.copy()

    y = y.map({"yes": 1, "no": 0})
    y = y.astype(int)

    print(f"  Shape: {X.shape}, Target distribution:\n{y.value_counts()}")

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, task="classification"
    )

    _save_split(X_train, y_train, "train", "bank_marketing")
    _save_split(X_val, y_val, "val", "bank_marketing")
    _save_split(X_test, y_test, "test", "bank_marketing")

    print("  Bank Marketing dataset saved.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_all_datasets():
    """
    Load, clean, and split all 5 datasets.
    Returns a dictionary mapping dataset name to (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    _ensure_dir(DATA_DIR)

    datasets = {}
    datasets["dry_bean"] = load_dry_bean()
    datasets["superconductivity"] = load_superconductivity()
    datasets["online_news"] = load_online_news()
    datasets["ai4i_maintenance"] = load_ai4i_maintenance()
    datasets["bank_marketing"] = load_bank_marketing()

    print("\n=== All datasets loaded and saved to data/ ===")
    return datasets


if __name__ == "__main__":
    load_all_datasets()
