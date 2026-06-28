"""
RUN 04: feature ablation and dimensionality reduction.

Purpose
-------
This script deepens the current feature-centric experiment:
1. compares market, volume, text and combined feature groups;
2. applies PCA to high-dimensional FinBERT embeddings using train only;
3. tunes Logistic Regression hyperparameter C and classification threshold on validation only;
4. evaluates the final selected configuration on test only.

The script is intentionally conservative: it does not tune on test and does not require heavy neural training.

Expected input
--------------
A single CSV file with at least:
- a date-like column: decision_date, date, Date, begin, or timestamp;
- either a ready target column y / target / label, or close/Close for target construction;
- price columns such as open, high, low, close, adj-close, inc-5, ...;
- optional volume column;
- FinBERT embedding columns, usually with prefixes finbert_, embedding_, emb_, or e_.

Example
-------
python experiments/run_04_feature_ablation.py \
  --input data/primary_dataset.csv \
  --output-dir artifacts/run_04 \
  --date-col decision_date \
  --target-col y
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DATE_CANDIDATES = ["decision_date", "date", "Date", "begin", "timestamp", "datetime"]
TARGET_CANDIDATES = ["y", "target", "label", "direction", "next_day_up"]
PRICE_CANDIDATES = ["open", "high", "low", "close", "adj-close", "adj_close", "adjclose"]
VOLUME_CANDIDATES = ["volume", "Volume", "vol", "turnover"]
EMBEDDING_PREFIXES = ["finbert_", "embedding_", "emb_", "e_"]


@dataclass
class RunConfig:
    input: str
    output_dir: str
    date_col: Optional[str]
    target_col: Optional[str]
    ticker_col: Optional[str]
    embedding_prefix: Optional[str]
    train_ratio: float
    val_ratio: float
    pca_components: Tuple[int, ...]
    c_grid: Tuple[float, ...]
    random_state: int


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="RUN 04 feature ablation experiment")
    parser.add_argument("--input", required=True, help="Path to dataset CSV")
    parser.add_argument("--output-dir", default="artifacts/run_04", help="Directory for outputs")
    parser.add_argument("--date-col", default=None, help="Date column. Auto-detected if omitted")
    parser.add_argument("--target-col", default=None, help="Target column. Auto-detected or built from close")
    parser.add_argument("--ticker-col", default=None, help="Optional ticker column for grouped target / previous baseline")
    parser.add_argument("--embedding-prefix", default=None, help="Prefix for FinBERT embedding columns")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--pca-components", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    return RunConfig(
        input=args.input,
        output_dir=args.output_dir,
        date_col=args.date_col,
        target_col=args.target_col,
        ticker_col=args.ticker_col,
        embedding_prefix=args.embedding_prefix,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pca_components=tuple(args.pca_components),
        c_grid=tuple(args.c_grid),
        random_state=args.random_state,
    )


def normalize_name(col: str) -> str:
    return str(col).strip()


def detect_column(df: pd.DataFrame, candidates: Iterable[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit column {explicit!r} not found")
        return explicit
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def ensure_target(df: pd.DataFrame, date_col: str, target_col: Optional[str], ticker_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    existing_target = detect_column(df, TARGET_CANDIDATES, target_col)
    if existing_target is not None:
        out = df.copy()
        out[existing_target] = out[existing_target].astype(int)
        return out.dropna(subset=[existing_target]).reset_index(drop=True), existing_target

    close_col = detect_column(df, ["close", "Close", "adj-close", "adj_close", "adjclose"])
    if close_col is None:
        raise ValueError("No target column and no close/adj-close column to build target")

    out = df.copy()
    group_cols = [ticker_col] if ticker_col and ticker_col in out.columns else []

    if group_cols:
        target_frames = []
        for _, g in out[[*group_cols, date_col, close_col]].drop_duplicates([*group_cols, date_col]).groupby(group_cols):
            g = g.sort_values(date_col).copy()
            g["y"] = (g[close_col].shift(-1) > g[close_col]).astype(float)
            target_frames.append(g[[*group_cols, date_col, "y"]])
        target_by_date = pd.concat(target_frames, ignore_index=True)
        out = out.merge(target_by_date, on=[*group_cols, date_col], how="left")
    else:
        target_by_date = out[[date_col, close_col]].drop_duplicates(date_col).sort_values(date_col).copy()
        target_by_date["y"] = (target_by_date[close_col].shift(-1) > target_by_date[close_col]).astype(float)
        out = out.merge(target_by_date[[date_col, "y"]], on=date_col, how="left")

    out = out.dropna(subset=["y"]).copy()
    out["y"] = out["y"].astype(int)
    return out.reset_index(drop=True), "y"


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def detect_embedding_columns(df: pd.DataFrame, explicit_prefix: Optional[str], exclude: Iterable[str]) -> List[str]:
    exclude_set = set(exclude)
    numeric_cols = [c for c in detect_numeric_columns(df) if c not in exclude_set]
    prefixes = [explicit_prefix] if explicit_prefix else EMBEDDING_PREFIXES
    cols: List[str] = []
    for col in numeric_cols:
        name = str(col)
        if any(name.startswith(prefix) for prefix in prefixes if prefix):
            cols.append(col)
    if len(cols) >= 8:
        return cols

    # Fallback: columns named 0..767 or emb numeric suffixes after a notebook export.
    numeric_name_cols = [c for c in numeric_cols if str(c).isdigit()]
    if len(numeric_name_cols) >= 32:
        return sorted(numeric_name_cols, key=lambda x: int(str(x)))
    return cols


def make_market_features(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = df.copy()
    groups: Dict[str, List[str]] = {}

    numeric_cols = detect_numeric_columns(out)
    price_raw = []
    for col in out.columns:
        name = str(col).lower()
        if name in PRICE_CANDIDATES or name.startswith("inc-") or name.startswith("inc_"):
            if col in numeric_cols:
                price_raw.append(col)
    groups["prices_raw"] = list(dict.fromkeys(price_raw))

    close_col = detect_column(out, ["close", "Close", "adj-close", "adj_close", "adjclose"])
    open_col = detect_column(out, ["open", "Open"])
    high_col = detect_column(out, ["high", "High"])
    low_col = detect_column(out, ["low", "Low"])
    volume_col = detect_column(out, VOLUME_CANDIDATES)

    date_level_cols = [c for c in [close_col, open_col, high_col, low_col, volume_col] if c]
    if date_level_cols:
        date_level = out[[date_col, *date_level_cols]].drop_duplicates(date_col).sort_values(date_col).copy()
        if close_col:
            date_level["log_return_1"] = np.log(date_level[close_col] / date_level[close_col].shift(1))
            for k in [2, 5, 10]:
                date_level[f"log_return_{k}"] = np.log(date_level[close_col] / date_level[close_col].shift(k))
                date_level[f"rolling_mean_return_{k}"] = date_level["log_return_1"].rolling(k).mean()
                date_level[f"rolling_vol_{k}"] = date_level["log_return_1"].rolling(k).std()
                date_level[f"ma_{k}"] = date_level[close_col].rolling(k).mean()
                date_level[f"resid_ma_{k}"] = date_level[close_col] - date_level[f"ma_{k}"]
        if close_col and open_col:
            date_level["close_to_open_return"] = np.log(date_level[close_col] / date_level[open_col])
        if high_col and low_col and close_col:
            date_level["high_low_range"] = (date_level[high_col] - date_level[low_col]) / date_level[close_col]
        if open_col and close_col:
            date_level["gap_return"] = np.log(date_level[open_col] / date_level[close_col].shift(1))
        if volume_col:
            date_level["log_volume"] = np.log1p(date_level[volume_col])
            for k in [5, 10, 20]:
                date_level[f"volume_ma_{k}"] = date_level[volume_col].rolling(k).mean()
                date_level[f"rel_volume_{k}"] = date_level[volume_col] / date_level[f"volume_ma_{k}"]
                vol_mean = date_level[volume_col].rolling(k).mean()
                vol_std = date_level[volume_col].rolling(k).std()
                date_level[f"volume_z_{k}"] = (date_level[volume_col] - vol_mean) / vol_std

        new_cols = [c for c in date_level.columns if c not in [date_col, *date_level_cols]]
        out = out.merge(date_level[[date_col, *new_cols]], on=date_col, how="left")

    returns_rolling = [
        c for c in out.columns
        if str(c).startswith(("log_return_", "rolling_mean_return_", "rolling_vol_", "ma_", "resid_ma_", "close_to_open_return", "high_low_range", "gap_return"))
        and pd.api.types.is_numeric_dtype(out[c])
    ]
    volume_rolling = [
        c for c in out.columns
        if str(c).startswith(("log_volume", "volume_ma_", "rel_volume_", "volume_z_"))
        and pd.api.types.is_numeric_dtype(out[c])
    ]
    groups["returns_rolling"] = returns_rolling
    groups["volume_rolling"] = volume_rolling
    groups["market_full"] = list(dict.fromkeys(price_raw + returns_rolling + volume_rolling))
    return out, groups


def add_text_meta_features(df: pd.DataFrame, date_col: str, embedding_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    meta_cols: List[str] = []
    if embedding_cols:
        out["embedding_norm"] = np.linalg.norm(out[embedding_cols].fillna(0).values, axis=1)
        meta_cols.append("embedding_norm")
    counts = out.groupby(date_col).size().rename("text_count").reset_index()
    counts["has_text"] = (counts["text_count"] > 0).astype(int)
    out = out.merge(counts, on=date_col, how="left")
    meta_cols.extend(["text_count", "has_text"])
    return out, meta_cols


def chronological_split(df: pd.DataFrame, date_col: str, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_dates = np.array(sorted(df[date_col].dropna().unique()))
    n_dates = len(unique_dates)
    if n_dates < 5:
        raise ValueError(f"Too few unique dates for chronological split: {n_dates}")
    n_train = max(1, int(math.floor(n_dates * train_ratio)))
    n_val = max(1, int(math.floor(n_dates * val_ratio)))
    if n_train + n_val >= n_dates:
        n_train = max(1, n_dates - 2)
        n_val = 1
    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train:n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val:])
    train_idx = df.index[df[date_col].isin(train_dates)].to_numpy()
    val_idx = df.index[df[date_col].isin(val_dates)].to_numpy()
    test_idx = df.index[df[date_col].isin(test_dates)].to_numpy()
    return train_idx, val_idx, test_idx


def clean_feature_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    x = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    return x


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_tau = 0.5
    best_score = -np.inf
    for tau in thresholds:
        pred = (proba >= tau).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau, float(best_score)


def safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def metrics_dict(y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, proba),
    }


def fit_logreg_group(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    c_grid: Tuple[float, ...],
    random_state: int,
) -> Tuple[Dict[str, float], Dict[str, int], pd.DataFrame, Dict[str, object]]:
    x = clean_feature_frame(df, feature_cols)
    y = df[target_col].astype(int).values

    # Fit imputation values and scaler only on train.
    train_median = x.iloc[train_idx].median(numeric_only=True)
    x_train = x.iloc[train_idx].fillna(train_median)
    x_val = x.iloc[val_idx].fillna(train_median)
    x_test = x.iloc[test_idx].fillna(train_median)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    best = None
    for c in c_grid:
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=random_state,
        )
        model.fit(x_train_s, y[train_idx])
        val_proba = model.predict_proba(x_val_s)[:, 1]
        tau, val_score = tune_threshold(y[val_idx], val_proba)
        if best is None or val_score > best["val_balanced_accuracy"]:
            best = {
                "model": model,
                "C": c,
                "threshold": tau,
                "val_balanced_accuracy": val_score,
            }

    model = best["model"]
    test_proba = model.predict_proba(x_test_s)[:, 1]
    test_pred = (test_proba >= best["threshold"]).astype(int)
    m = metrics_dict(y[test_idx], test_pred, test_proba)
    cm = confusion_matrix(y[test_idx], test_pred, labels=[0, 1])
    cm_dict = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}

    preds = pd.DataFrame({
        "row_index": test_idx,
        "date": df.iloc[test_idx].iloc[:, df.columns.get_loc(df.attrs["date_col"])].values,
        "y_true": y[test_idx],
        "y_proba": test_proba,
        "y_pred": test_pred,
    })
    details = {
        "selected_C": best["C"],
        "selected_threshold": best["threshold"],
        "val_balanced_accuracy": best["val_balanced_accuracy"],
        "n_features": len(feature_cols),
    }
    return m, cm_dict, preds, details


def evaluate_majority(df: pd.DataFrame, target_col: str, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int], pd.DataFrame, Dict[str, object]]:
    y = df[target_col].astype(int).values
    majority = int(pd.Series(y[train_idx]).mode().iloc[0])
    pred = np.full(len(test_idx), majority, dtype=int)
    proba = np.full(len(test_idx), float(majority), dtype=float)
    m = metrics_dict(y[test_idx], pred, proba)
    cm = confusion_matrix(y[test_idx], pred, labels=[0, 1])
    cm_dict = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    preds = pd.DataFrame({"row_index": test_idx, "y_true": y[test_idx], "y_proba": proba, "y_pred": pred})
    return m, cm_dict, preds, {"majority_class": majority, "n_features": 0}


def evaluate_previous_direction(df: pd.DataFrame, date_col: str, target_col: str, test_idx: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int], pd.DataFrame, Dict[str, object]]:
    y_by_date = df[[date_col, target_col]].drop_duplicates(date_col).sort_values(date_col).copy()
    y_by_date["previous_direction"] = y_by_date[target_col].shift(1)
    temp = df[[date_col, target_col]].merge(y_by_date[[date_col, "previous_direction"]], on=date_col, how="left")
    test_temp = temp.iloc[test_idx].dropna(subset=["previous_direction"]).copy()
    y_true = test_temp[target_col].astype(int).values
    pred = test_temp["previous_direction"].astype(int).values
    proba = pred.astype(float)
    m = metrics_dict(y_true, pred, proba)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    cm_dict = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    preds = pd.DataFrame({"row_index": test_temp.index, "y_true": y_true, "y_proba": proba, "y_pred": pred})
    return m, cm_dict, preds, {"n_features": 0}


def make_pca_features(
    df: pd.DataFrame,
    embedding_cols: List[str],
    train_idx: np.ndarray,
    all_indices: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    x = clean_feature_frame(df, embedding_cols)
    train_median = x.iloc[train_idx].median(numeric_only=True)
    x_train = x.iloc[train_idx].fillna(train_median)
    x_all = x.iloc[all_indices].fillna(train_median)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_all_s = scaler.transform(x_all)

    n = min(n_components, x_train_s.shape[1], max(1, x_train_s.shape[0] - 1))
    pca = PCA(n_components=n, random_state=random_state)
    pca.fit(x_train_s)
    transformed = pca.transform(x_all_s)
    cols = [f"finbert_pca_{n}_{i + 1}" for i in range(n)]
    pca_df = pd.DataFrame(transformed, columns=cols, index=all_indices)
    info = {
        "requested_components": n_components,
        "actual_components": n,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return pca_df, cols, info


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.input)
    df.columns = [normalize_name(c) for c in df.columns]
    date_col = detect_column(df, DATE_CANDIDATES, config.date_col)
    if date_col is None:
        raise ValueError(f"Could not detect date column. Candidates: {DATE_CANDIDATES}")
    df = coerce_dates(df, date_col)
    df, target_col = ensure_target(df, date_col, config.target_col, config.ticker_col)
    df.attrs["date_col"] = date_col

    df, market_groups = make_market_features(df, date_col)
    exclude_cols = [date_col, target_col]
    if config.ticker_col:
        exclude_cols.append(config.ticker_col)
    embedding_cols = detect_embedding_columns(df, config.embedding_prefix, exclude_cols)
    df, text_meta_cols = add_text_meta_features(df, date_col, embedding_cols)

    # Remove rows with missing target and sort again after feature engineering.
    df = df.sort_values(date_col).reset_index(drop=True)
    df.attrs["date_col"] = date_col
    train_idx, val_idx, test_idx = chronological_split(df, date_col, config.train_ratio, config.val_ratio)
    all_idx = df.index.to_numpy()

    feature_groups: Dict[str, List[str]] = {
        "prices_raw": market_groups.get("prices_raw", []),
        "returns_rolling": market_groups.get("returns_rolling", []),
        "volume_rolling": market_groups.get("volume_rolling", []),
        "market_full": market_groups.get("market_full", []),
        "finbert_full": embedding_cols,
        "text_meta": text_meta_cols,
    }
    feature_groups["market_full_text_meta"] = list(dict.fromkeys(feature_groups["market_full"] + text_meta_cols))

    pca_info: Dict[str, Dict[str, object]] = {}
    for k in config.pca_components:
        if embedding_cols:
            pca_df, pca_cols, info = make_pca_features(df, embedding_cols, train_idx, all_idx, k, config.random_state)
            for col in pca_cols:
                df[col] = pca_df[col]
            pca_info[f"finbert_pca_{k}"] = info
            feature_groups[f"finbert_pca_{k}"] = pca_cols
            feature_groups[f"market_full_finbert_pca_{k}"] = list(dict.fromkeys(feature_groups["market_full"] + pca_cols))
            feature_groups[f"market_full_finbert_pca_{k}_text_meta"] = list(dict.fromkeys(feature_groups["market_full"] + pca_cols + text_meta_cols))

    metrics_rows = []
    cm_rows = []
    prediction_frames = []
    selection_rows = []

    # Baselines.
    for group_name, evaluator in [
        ("majority", lambda: evaluate_majority(df, target_col, train_idx, test_idx)),
        ("previous_direction", lambda: evaluate_previous_direction(df, date_col, target_col, test_idx)),
    ]:
        m, cm, preds, details = evaluator()
        metrics_rows.append({"model": "baseline", "feature_group": group_name, **m, **details})
        cm_rows.append({"model": "baseline", "feature_group": group_name, **cm})
        preds["model"] = "baseline"
        preds["feature_group"] = group_name
        prediction_frames.append(preds)

    # Logistic Regression feature groups.
    for group_name, cols in feature_groups.items():
        cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            continue
        m, cm, preds, details = fit_logreg_group(
            df=df,
            feature_cols=cols,
            target_col=target_col,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            c_grid=config.c_grid,
            random_state=config.random_state,
        )
        metrics_rows.append({"model": "LogisticRegression", "feature_group": group_name, **m, **details})
        cm_rows.append({"model": "LogisticRegression", "feature_group": group_name, **cm})
        selection_rows.append({"feature_group": group_name, **details})
        preds["model"] = "LogisticRegression"
        preds["feature_group"] = group_name
        prediction_frames.append(preds)

    metrics = pd.DataFrame(metrics_rows).sort_values(["balanced_accuracy", "accuracy"], ascending=False)
    confusion = pd.DataFrame(cm_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv", index=False)
    selections.to_csv(output_dir / "validation_selection.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    feature_manifest = {
        name: cols for name, cols in feature_groups.items()
    }
    with open(output_dir / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, ensure_ascii=False, indent=2, default=str)

    summary = {
        "config": asdict(config),
        "date_col": date_col,
        "target_col": target_col,
        "n_observations": int(len(df)),
        "n_unique_dates": int(df[date_col].nunique()),
        "date_min": str(df[date_col].min().date()),
        "date_max": str(df[date_col].max().date()),
        "train_size": int(len(train_idx)),
        "validation_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_date_min": str(df.iloc[train_idx][date_col].min().date()),
        "train_date_max": str(df.iloc[train_idx][date_col].max().date()),
        "validation_date_min": str(df.iloc[val_idx][date_col].min().date()),
        "validation_date_max": str(df.iloc[val_idx][date_col].max().date()),
        "test_date_min": str(df.iloc[test_idx][date_col].min().date()),
        "test_date_max": str(df.iloc[test_idx][date_col].max().date()),
        "n_embedding_cols": int(len(embedding_cols)),
        "embedding_cols_sample": list(map(str, embedding_cols[:10])),
        "pca_info": pca_info,
        "feature_group_sizes": {name: len(cols) for name, cols in feature_groups.items()},
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("RUN 04 completed")
    print(f"Output directory: {output_dir}")
    print(metrics.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
