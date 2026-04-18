"""
Train MLP and TabNet on all 5 datasets.

Preprocessing is loaded from saved preprocessor.pkl artefacts produced by
train_trees_and_xrfm.py, guaranteeing identical feature transformations across
all models (no re-fitting on val/test).

Locked hyper-parameters (NOT tunable per .cursorrules):
  MLP   : hidden_layer_sizes=(128,64), alpha=0.001, max_iter=200, early_stopping=True
  TabNet: n_d=8, n_a=8, n_steps=3, gamma=1.3, max_epochs=50, patience=10,
          batch_size=1024, virtual_batch_size=128  +  eval_set triggers early stopping

Saved artefacts
---------------
saved_models/<dataset_name>/mlp.pkl          (sklearn joblib)
saved_models/<dataset_name>/tabnet/          (pytorch-tabnet native .zip)

Training times are appended to results/training_times_dl.csv.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier, MLPRegressor

RANDOM_STATE     = 42
MAX_TRAIN_SAMPLES = None  

ROOT_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
RES_DIR   = os.path.join(ROOT_DIR, "results")

DATASET_CONFIG = {
    "dry_bean":          {"task": "multiclass"},
    "ai4i_maintenance":  {"task": "binary"},
    "bank_marketing":    {"task": "binary"},
    "online_news":       {"task": "regression"},
    "superconductivity": {"task": "regression"},
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_split(dataset_name, split):
    base = os.path.join(DATA_DIR, dataset_name, split)
    X = pd.read_csv(os.path.join(base, "X.csv"))
    y = pd.read_csv(os.path.join(base, "y.csv")).iloc[:, 0]
    return X, y


# ── Preprocessing (reuse saved artefacts) ────────────────────────────────────

def apply_preprocessor(meta, X_train_raw, X_val_raw, X_test_raw):
    """
    Apply the already-fitted scaler and OHE from preprocessor.pkl.
    No re-fitting occurs here – strictly transform only.
    """
    scaler   = meta["scaler"]
    ohe      = meta["ohe"]
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    def _transform(X_raw):
        parts = []
        if num_cols:
            parts.append(scaler.transform(X_raw[num_cols]))
        if cat_cols:
            parts.append(ohe.transform(X_raw[cat_cols]))
        if not parts:
            return np.empty((len(X_raw), 0), dtype=np.float32)
        return np.hstack(parts).astype(np.float32)

    return _transform(X_train_raw), _transform(X_val_raw), _transform(X_test_raw)


# ── MLP ───────────────────────────────────────────────────────────────────────

def train_mlp(X_train, y_train, task, seed=RANDOM_STATE):
    """Locked config – no grid search allowed."""
    common = dict(
        hidden_layer_sizes=(128, 64),
        alpha=0.001,
        max_iter=200,
        early_stopping=True,
        random_state=seed,
    )
    model = MLPRegressor(**common) if task == "regression" else MLPClassifier(**common)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return model, train_time


# ── TabNet ────────────────────────────────────────────────────────────────────

def train_tabnet(X_train, y_train, X_val, y_val, task, seed=RANDOM_STATE):
    """
    Locked config – no grid search allowed.
    eval_set is mandatory to trigger early stopping (patience=10).
    """
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    tabnet_common = dict(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        seed=seed,
        verbose=0,
    )
    fit_common = dict(
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
    )

    if task == "regression":
        model = TabNetRegressor(**tabnet_common)
        y_tr  = y_train.reshape(-1, 1).astype(np.float32)
        y_vl  = y_val.reshape(-1, 1).astype(np.float32)
        eval_metric = ["rmse"]
    else:
        model = TabNetClassifier(**tabnet_common)
        y_tr  = y_train.astype(int)
        y_vl  = y_val.astype(int)
        eval_metric = ["accuracy"]

    t0 = time.perf_counter()
    model.fit(
        X_train, y_tr,
        eval_set=[(X_val, y_vl)],
        eval_metric=eval_metric,
        **fit_common,
    )
    train_time = time.perf_counter() - t0
    return model, train_time


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_all():
    _ensure_dir(RES_DIR)
    records = []
    smoke   = MAX_TRAIN_SAMPLES is not None
    suffix  = "_smoke" if smoke else ""

    for dataset_name, cfg in DATASET_CONFIG.items():
        split_path = os.path.join(DATA_DIR, dataset_name, "train")
        prep_path  = os.path.join(MODEL_DIR, dataset_name, "preprocessor.pkl")

        if not os.path.exists(split_path):
            print(f"[SKIP] {dataset_name} – data not found. Run data_loader.py first.")
            continue
        if not os.path.exists(prep_path):
            print(f"[SKIP] {dataset_name} – preprocessor.pkl missing. "
                  "Run train_trees_and_xrfm.py first.")
            continue

        task = cfg["task"]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  |  Task: {task}")
        print('='*60)

        # ── Load raw splits ───────────────────────────────────────────────────
        X_train_raw, y_train = _load_split(dataset_name, "train")
        X_val_raw,   y_val   = _load_split(dataset_name, "val")
        X_test_raw,  y_test  = _load_split(dataset_name, "test")

        # ── Optional sample cap ───────────────────────────────────────────────
        if smoke and len(X_train_raw) > MAX_TRAIN_SAMPLES:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X_train_raw), MAX_TRAIN_SAMPLES, replace=False)
            X_train_raw = X_train_raw.iloc[idx].reset_index(drop=True)
            y_train     = y_train.iloc[idx].reset_index(drop=True)
            print(f"  [SMOKE TEST] Training capped at {MAX_TRAIN_SAMPLES} samples.")

        # ── Apply saved preprocessor (no re-fit) ─────────────────────────────
        meta = joblib.load(prep_path)
        X_train, X_val, X_test = apply_preprocessor(
            meta, X_train_raw, X_val_raw, X_test_raw
        )
        print(f"  X_train shape: {X_train.shape}")

        # ── Label encoding for multiclass ─────────────────────────────────────
        le_path = os.path.join(MODEL_DIR, dataset_name, "label_encoder.pkl")
        if task == "multiclass" and os.path.exists(le_path):
            le      = joblib.load(le_path)
            y_train = le.transform(y_train)
            y_val   = le.transform(y_val)
            y_test  = le.transform(y_test)

        y_train_np = np.asarray(y_train)
        y_val_np   = np.asarray(y_val)

        model_dir = os.path.join(MODEL_DIR, dataset_name)
        _ensure_dir(model_dir)

        # ── MLP ───────────────────────────────────────────────────────────────
        print("  Training MLP...", end=" ", flush=True)
        mlp_model, mlp_time = train_mlp(X_train, y_train_np, task)
        joblib.dump(mlp_model, os.path.join(model_dir, f"mlp{suffix}.pkl"))
        print(f"done  ({mlp_time:.1f}s)  |  "
              f"actual_iters={mlp_model.n_iter_}")
        records.append({"dataset": dataset_name, "model": "MLP",
                        "task": task, "train_time_s": round(mlp_time, 3)})

        # ── TabNet ────────────────────────────────────────────────────────────
        print("  Training TabNet...", end=" ", flush=True)
        try:
            tabnet_model, tabnet_time = train_tabnet(
                X_train, y_train_np, X_val, y_val_np, task
            )
            tabnet_dir = os.path.join(model_dir, f"tabnet{suffix}")
            _ensure_dir(tabnet_dir)
            tabnet_model.save_model(os.path.join(tabnet_dir, "model"))
            print(f"done  ({tabnet_time:.1f}s)")
            records.append({"dataset": dataset_name, "model": "TabNet",
                            "task": task, "train_time_s": round(tabnet_time, 3)})
        except Exception as e:
            print(f"FAILED – {e}")

    # ── Save training time summary ─────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        out_path = os.path.join(RES_DIR, "training_times_dl.csv")
        df.to_csv(out_path, index=False)
        print(f"\n Training times saved → {out_path}")
        print(df.to_string(index=False))

    print("\n=== Done: MLP / TabNet training complete ===")


if __name__ == "__main__":
    run_all()
