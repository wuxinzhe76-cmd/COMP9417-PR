"""
Train xRFM, XGBoost, and Random Forest on all 5 datasets.

Preprocessing is done here (StandardScaler + OneHotEncoder) rather than in
data_loader.py to strictly enforce the no-leakage rule: all transformers are
fit ONLY on X_train, then applied to X_val and X_test.

Saved artefacts
---------------
saved_models/<dataset_name>/xrfm.pkl
saved_models/<dataset_name>/xgb.pkl
saved_models/<dataset_name>/rf.pkl
saved_models/<dataset_name>/label_encoder.pkl   (only for multiclass tasks)
saved_models/<dataset_name>/preprocessor.pkl    (scaler + ohe + meta)

Training times are appended to results/training_times.csv.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = None

ROOT_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
RES_DIR   = os.path.join(ROOT_DIR, "results")

# ── Dataset registry ─────────────────────────────────────────────────────────
# task: "regression" | "binary" | "multiclass"
DATASET_CONFIG = {
    "dry_bean":          {"task": "multiclass"},
    "ai4i_maintenance":  {"task": "binary"},
    "bank_marketing":    {"task": "binary"},
    "online_news":       {"task": "regression"},
    "superconductivity": {"task": "regression"},
}


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load_split(dataset_name, split):
    base = os.path.join(DATA_DIR, dataset_name, split)
    X = pd.read_csv(os.path.join(base, "X.csv"))
    y = pd.read_csv(os.path.join(base, "y.csv")).iloc[:, 0]
    return X, y


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X_train, X_val, X_test):
    """
    Fit StandardScaler on numerical columns and OneHotEncoder on categorical
    columns using ONLY X_train. Transform val and test with the fitted objects.

    Returns
    -------
    X_train_np, X_val_np, X_test_np : np.ndarray  (float32)
    categorical_info                : dict  (for xRFM)
    meta                            : dict  (scaler, ohe, col names, n_num)
    """
    num_cols = X_train.select_dtypes(include=["int64", "float64",
                                               "int32",  "float32"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    scaler = StandardScaler()
    ohe    = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if num_cols:
        X_num_train = scaler.fit_transform(X_train[num_cols])
        X_num_val   = scaler.transform(X_val[num_cols])
        X_num_test  = scaler.transform(X_test[num_cols])
    else:
        X_num_train = np.empty((len(X_train), 0))
        X_num_val   = np.empty((len(X_val),   0))
        X_num_test  = np.empty((len(X_test),  0))

    if cat_cols:
        X_cat_train = ohe.fit_transform(X_train[cat_cols])
        X_cat_val   = ohe.transform(X_val[cat_cols])
        X_cat_test  = ohe.transform(X_test[cat_cols])
    else:
        X_cat_train = np.empty((len(X_train), 0))
        X_cat_val   = np.empty((len(X_val),   0))
        X_cat_test  = np.empty((len(X_test),  0))

    X_train_np = np.hstack([X_num_train, X_cat_train]).astype(np.float32)
    X_val_np   = np.hstack([X_num_val,   X_cat_val  ]).astype(np.float32)
    X_test_np  = np.hstack([X_num_test,  X_cat_test ]).astype(np.float32)

    # ── Build xRFM categorical_info ──────────────────────────────────────────
    n_num = X_num_train.shape[1]
    categorical_indices = []
    categorical_vectors = []
    start = n_num
    for cats in getattr(ohe, "categories_", []):
        cat_len = len(cats)
        idxs = torch.arange(start, start + cat_len, dtype=torch.long)
        categorical_indices.append(idxs)
        categorical_vectors.append(torch.eye(cat_len, dtype=torch.float32))
        start += cat_len

    numerical_indices = torch.arange(0, n_num, dtype=torch.long)
    categorical_info = {
        "numerical_indices":   numerical_indices,
        "categorical_indices": categorical_indices,
        "categorical_vectors": categorical_vectors,
    }

    meta = {
        "scaler":   scaler,
        "ohe":      ohe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_num":    n_num,
    }
    return X_train_np, X_val_np, X_test_np, categorical_info, meta


# ── xRFM ─────────────────────────────────────────────────────────────────────

def train_xrfm(X_train, y_train, X_val, y_val, task, categorical_info, seed=RANDOM_STATE):
    from xrfm import xRFM  # lazy import so script still loads without xrfm

    # xRFM's "accuracy" tuning metric has an internal dimension mismatch bug
    # when validation set refilling is triggered. Using "mse" universally is
    # safe: for classification with 0/1 or 0-K integer labels, MSE-based
    # splitting still learns the correct decision boundaries at each leaf.
    tuning_metric = "mse"
    rfm_params = {
        "model": {
            "kernel":         "l2",
            "bandwidth":      10.0,
            "exponent":       1.0,
            "diag":           False,
            "bandwidth_mode": "constant",
        },
        "fit": {
            "reg":            1e-3,
            "iters":          3,
            "verbose":        False,
            "early_stop_rfm": True,
        },
    }
    model = xRFM(
        rfm_params=rfm_params,
        tuning_metric=tuning_metric,
        categorical_info=categorical_info,
        max_leaf_size=3000,
        n_trees=1,
        random_state=seed,
        verbose=False,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train, np.asarray(y_train).astype(np.float32),
        X_val,   np.asarray(y_val).astype(np.float32),
    )
    train_time = time.perf_counter() - t0
    return model, train_time


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgb(X_train, y_train, X_val, y_val, task, n_classes=None, seed=RANDOM_STATE):
    common = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
    )
    if task == "regression":
        model = XGBRegressor(**common, eval_metric="rmse")
    elif task == "binary":
        model = XGBClassifier(**common, eval_metric="logloss")
    else:  # multiclass
        model = XGBClassifier(
            **common,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
        )

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.perf_counter() - t0
    return model, train_time


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_rf(X_train, y_train, task, seed=RANDOM_STATE):
    if task == "regression":
        model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return model, train_time


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_all():
    _ensure_dir(RES_DIR)
    records = []

    for dataset_name, cfg in DATASET_CONFIG.items():
        split_path = os.path.join(DATA_DIR, dataset_name, "train")
        if not os.path.exists(split_path):
            print(f"[SKIP] {dataset_name} – data not found, run data_loader.py first.")
            continue

        task = cfg["task"]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  |  Task: {task}")
        print('='*60)

        # ── Load raw splits ──────────────────────────────────────────────────
        X_train_raw, y_train = _load_split(dataset_name, "train")
        X_val_raw,   y_val   = _load_split(dataset_name, "val")
        X_test_raw,  y_test  = _load_split(dataset_name, "test")

        # ── Optional sample cap (for smoke-test runs) ────────────────────────
        if MAX_TRAIN_SAMPLES is not None and len(X_train_raw) > MAX_TRAIN_SAMPLES:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X_train_raw), MAX_TRAIN_SAMPLES, replace=False)
            X_train_raw = X_train_raw.iloc[idx].reset_index(drop=True)
            y_train     = y_train.iloc[idx].reset_index(drop=True)
            print(f"  [SMOKE TEST] Training capped at {MAX_TRAIN_SAMPLES} samples.")

        # ── Label encoding (multiclass string labels → 0-indexed int) ────────
        le = None
        if task == "multiclass" and y_train.dtype == object:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val   = le.transform(y_val)
            y_test  = le.transform(y_test)
            print(f"  LabelEncoder: {list(le.classes_)}")

        n_classes = int(y_train.max()) + 1 if task == "multiclass" else None

        # ── Preprocess ───────────────────────────────────────────────────────
        X_train, X_val, X_test, cat_info, meta = preprocess(
            X_train_raw, X_val_raw, X_test_raw
        )
        print(f"  X_train shape: {X_train.shape}  "
              f"(num={meta['n_num']}, cat_ohe={X_train.shape[1]-meta['n_num']})")

        # ── Save preprocessor + label encoder ────────────────────────────────
        model_dir = os.path.join(MODEL_DIR, dataset_name)
        _ensure_dir(model_dir)
        joblib.dump(meta, os.path.join(model_dir, "preprocessor.pkl"))
        if le is not None:
            joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

        smoke = MAX_TRAIN_SAMPLES is not None
        suffix = "_smoke" if smoke else ""

        # ── xRFM ─────────────────────────────────────────────────────────────
        xrfm_path = os.path.join(model_dir, f"xrfm{suffix}.pkl")
        print("  Training xRFM...", end=" ", flush=True)
        try:
            xrfm_model, xrfm_time = train_xrfm(
                X_train, y_train, X_val, y_val, task, cat_info
            )
            joblib.dump(xrfm_model, xrfm_path)
            print(f"done  ({xrfm_time:.1f}s)")
            records.append({"dataset": dataset_name, "model": "xRFM",
                            "task": task, "train_time_s": round(xrfm_time, 3)})
        except Exception as e:
            print(f"FAILED – {e}")

        # ── XGBoost ──────────────────────────────────────────────────────────
        xgb_path = os.path.join(model_dir, f"xgb{suffix}.pkl")
        print("  Training XGBoost...", end=" ", flush=True)
        xgb_model, xgb_time = train_xgb(
            X_train, y_train, X_val, y_val, task, n_classes
        )
        joblib.dump(xgb_model, xgb_path)
        print(f"done  ({xgb_time:.1f}s)")
        records.append({"dataset": dataset_name, "model": "XGBoost",
                        "task": task, "train_time_s": round(xgb_time, 3)})

        # ── Random Forest ────────────────────────────────────────────────────
        rf_path = os.path.join(model_dir, f"rf{suffix}.pkl")
        print("  Training Random Forest...", end=" ", flush=True)
        rf_model, rf_time = train_rf(X_train, y_train, task)
        joblib.dump(rf_model, rf_path)
        print(f"done  ({rf_time:.1f}s)")
        records.append({"dataset": dataset_name, "model": "RF",
                        "task": task, "train_time_s": round(rf_time, 3)})

    # ── Save training time summary ────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        out_path = os.path.join(RES_DIR, "training_times.csv")
        df.to_csv(out_path, index=False)
        print(f"\n Training times saved → {out_path}")
        print(df.to_string(index=False))

    print("\n=== Done: xRFM / XGBoost / RF training complete ===")


if __name__ == "__main__":
    run_all()
