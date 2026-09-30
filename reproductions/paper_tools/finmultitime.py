"""Explicit reconstruction of FinMultiTime recurrent/CNN baselines; not author code."""
import argparse
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare import LOCK, sha256, verify

MODELS = ("RNN", "LSTM", "GRU", "CNN", "TimeNetReconstructed")


def read_price_archive(published, ticker):
    spec = json.loads(LOCK.read_text())["finmultitime"]
    item = next(i for i in spec["files"] if i["path"] == "time_series/HS300_time_series.zip")
    archive = Path(published) / item["path"]
    verify(archive, item)
    if not ticker or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-' for c in ticker):
        raise ValueError("Invalid ticker")
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if n.endswith('/' + ticker + '.csv')]
        if len(matches) != 1:
            raise ValueError(f"Ticker must identify one CSV, found {len(matches)}")
        with z.open(matches[0]) as f:
            frame = pd.read_csv(f)
    frame = frame.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
    # Preserve local market day; converting to UTC would shift midnight to previous day.
    frame["date"] = pd.to_datetime(frame.date.str.slice(0, 10), format="%Y-%m-%d", errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.date.duplicated().any() or not np.isfinite(frame[["open", "high", "low", "close"]]).all().all():
        raise ValueError("Duplicate dates or nonfinite OHLC")
    return frame, sha256(archive)


def join_available_features(price, features, columns):
    """Explicit end-of-day availability; no backfill and no inferred filing release dates."""
    if not columns or any(c not in features for c in ["available_at", *columns]):
        raise ValueError("Require available_at plus explicit numeric feature columns")
    right = features[["available_at", *columns]].copy()
    right["available_at"] = pd.to_datetime(right.available_at, utc=True)
    if right.available_at.duplicated().any() or not np.isfinite(right[columns]).all().all():
        raise ValueError("Duplicate availability or nonfinite features")
    left = price.copy()
    # This adapter is HS300-specific, not a US market/calendar approximation.
    left["cutoff"] = (left.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).dt.tz_convert("UTC")
    joined = pd.merge_asof(left.sort_values("cutoff"), right.sort_values("available_at"),
                           left_on="cutoff", right_on="available_at", direction="backward")
    joined[columns] = joined[columns].fillna(0.0)
    return joined


class Windows(Dataset):
    def __init__(self, values, targets, starts, history, horizon):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.starts, self.history, self.horizon = list(starts), history, horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        return self.values[start-self.history:start], self.targets[start:start+self.horizon]


def split_windows(dates, history, horizon, train_end, val_end):
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_end >= val_end or history < 1 or horizon < 1:
        raise ValueError("Invalid chronological boundaries/history/horizon")
    result = {"train": [], "val": [], "test": []}
    for start in range(history, len(dates)-horizon+1):
        first, last = dates.iloc[start], dates.iloc[start+horizon-1]
        if last < train_end:
            result["train"].append(start)
        elif first >= train_end and last < val_end:
            result["val"].append(start)
        elif first >= val_end:
            result["test"].append(start)
    if any(not v for v in result.values()):
        raise ValueError("Empty split: adjust explicit dates or data coverage")
    return result


class Forecast(nn.Module):
    def __init__(self, model, inputs, history, horizon, hidden=64):
        super().__init__()
        self.horizon = horizon
        if model in ("RNN", "LSTM", "GRU"):
            self.encoder = getattr(nn, model)(inputs, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, horizon*4)
            self.recurrent = True
        elif model in ("CNN", "TimeNetReconstructed"):
            depth = 2 if model == "CNN" else 4
            layers = []
            for i in range(depth):
                layers.extend([nn.Conv1d(inputs if i == 0 else hidden, hidden, 3, padding=1), nn.ReLU()])
            self.encoder = nn.Sequential(*layers)
            self.head = nn.Linear(hidden*history, horizon*4)
            self.recurrent = False
        else:
            raise ValueError(model)

    def forward(self, x):
        if self.recurrent:
            encoded, _ = self.encoder(x)
            encoded = encoded[:, -1]
        else:
            encoded = self.encoder(x.transpose(1, 2)).flatten(1)
        return self.head(encoded).reshape(-1, self.horizon, 4)


def evaluate(model, loader):
    model.eval()
    preds, targets, bases = [], [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x))
            targets.append(y)
            bases.append(x[:, -1:, :4].expand_as(y))
    return tuple(torch.cat(v).numpy() for v in (preds, targets, bases))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--published", type=Path, required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--train-end", required=True, help="Exclusive train cutoff YYYY-MM-DD")
    p.add_argument("--val-end", required=True, help="Exclusive validation cutoff YYYY-MM-DD")
    p.add_argument("--history", type=int, default=96)
    p.add_argument("--horizon", type=int, choices=[24, 48, 96], default=24)
    p.add_argument("--model", choices=MODELS, default="GRU")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--features", type=Path)
    p.add_argument("--feature-columns", nargs="+", default=[])
    p.add_argument("--modality", choices=["price", "news", "table", "image"], default="price")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke-batches", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.smoke_batches < 0:
        p.error("Invalid training parameters")
    if (args.modality != "price") != bool(args.features) or bool(args.features) != bool(args.feature_columns):
        p.error("Non-price modality requires features with available_at and explicit columns")
    frame, archive_hash = read_price_archive(args.published, args.ticker)
    feature_hash = None
    if args.features:
        feature_hash = sha256(args.features)
        frame = join_available_features(frame, pd.read_csv(args.features), args.feature_columns)
    columns = ["open", "high", "low", "close", *args.feature_columns]
    if len(set(columns)) != len(columns):
        p.error("Feature columns must not collide with OHLC")
    starts = split_windows(frame.date, args.history, args.horizon, args.train_end, args.val_end)
    train = frame[frame.date < pd.Timestamp(args.train_end)][columns].to_numpy(dtype=np.float64)
    mean, scale = train.mean(0), train.std(0)
    scale[scale < 1e-12] = 1
    values = (frame[columns].to_numpy(dtype=np.float64)-mean)/scale
    manifest = {"protocol": "reconstructed-FinMultiTime-baselines", "archive_sha256": archive_hash,
                "features_sha256": feature_hash, "config": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "windows": {k: len(v) for k,v in starts.items()}, "mean": mean.tolist(), "scale": scale.tolist(),
                "smoke_only": bool(args.smoke_batches), "limitations": ["Author architectures and exact splits unavailable",
                "Horizons count trading rows, not hours", "No author outlier removal", "Only HS300 adapter implemented",
                "TimeNetReconstructed is a four-layer CNN hypothesis, not recovered author TimeNet",
                "Feature availability must be independently verified; timestamp join alone cannot establish provenance"]}
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    loaders = {}
    for split in starts:
        ds = Windows(values, values[:, :4], starts[split], args.history, args.horizon)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train")
        if args.smoke_batches:
            from itertools import islice
            loader = list(islice(loader, args.smoke_batches))
        loaders[split] = loader
    model = Forecast(args.model, len(columns), args.history, args.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = ((model(x)-y)**2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        pred, target, _ = evaluate(model, loaders["val"])
        val = float(np.mean((pred-target)**2))
        if not np.isfinite([*losses, val]).all():
            raise ValueError("Nonfinite loss")
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val})
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best, stale = val, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "mean": mean.tolist(), "scale": scale.tolist(),
                        "columns": columns, "config": manifest["config"]}, args.output / "checkpoint.pt")
        else:
            stale += 1
            if stale >= 5:
                break
    saved = torch.load(args.output / "checkpoint.pt", weights_only=True)
    model.load_state_dict(saved["model"])
    pred, target, baseline = evaluate(model, loaders["test"])
    if not all(np.isfinite(v).all() for v in (pred, target, baseline)):
        raise ValueError("Nonfinite test predictions or targets")
    metrics = {"scaled_mse": float(np.mean((pred-target)**2)), "scaled_mae": float(np.mean(abs(pred-target))),
               "raw_mse": float(np.mean(((pred-target)*scale[:4])**2)),
               "raw_mae": float(np.mean(abs((pred-target)*scale[:4]))),
               "persistence_scaled_mse": float(np.mean((baseline-target)**2)), "test_windows": len(target),
               "selected_epoch": saved["epoch"], "smoke_only": bool(args.smoke_batches)}
    np.savez_compressed(args.output / "predictions.npz", predictions=pred, targets=target, persistence=baseline)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "SUCCESS.json").write_text(json.dumps({"checkpoint_sha256": sha256(args.output / "checkpoint.pt"), "metrics": metrics}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
