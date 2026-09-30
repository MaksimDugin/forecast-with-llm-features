"""Checkpointed orchestration around unchanged pinned FinTexTS model/training code."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare import LOCK, sha256, verify_source
from run_fintexts import MODELS, PREFIXES, inspect_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="DLinear")
    parser.add_argument("--prefixes", nargs="+", choices=PREFIXES, default=list(PREFIXES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-weight", type=float, default=0.1)
    parser.add_argument("--scaler", choices=["author-full", "train-only"], default="author-full")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--portable-autoformer", action="store_true", help="Opt-in in-memory device correction")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.smoke_batches < 0 or not 0 <= args.text_weight <= 1:
        parser.error("Invalid training parameters")
    source = verify_source(args.author, json.loads(LOCK.read_text())["fintexts"])
    info = inspect_parquet(args.data, args.prefixes, 64, 3, args.batch_size)
    plan = {"protocol": "author-models-checkpointed", "source_commit": json.loads(LOCK.read_text())["fintexts"]["author_commit"],
            "input_sha256": sha256(args.data), "preflight": info, "config": vars(args).copy(),
            "smoke_only": bool(args.smoke_batches), "limitations": ["Encoder identity must be checked in embedding manifest",
            "author-full scaler sees validation and test; train-only is a changed protocol"]}
    plan["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in plan["config"].items()}
    embedding_manifest = args.data.parent / "embedding_manifest.json"
    plan["embedding_manifest"] = {"sha256": sha256(embedding_manifest), "content": json.loads(embedding_manifest.read_text())} if embedding_manifest.is_file() else None
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(plan, indent=2))
    sys.path[:0] = [str(source), str(source / "forecasting_task")]
    runner = importlib.import_module("forecasting_task.run")
    if args.portable_autoformer:
        from author_compat import apply_device_fix
        print("COMPATIBILITY_PATCH =", apply_device_fix(), flush=True)
    # Reuse all author architecture defaults without changing the checkout.
    sys.argv = ["author-run", "--model_type", args.model, "--seed", str(args.seed),
                "--data_path", str(args.data.resolve()), "--batch_size", str(args.batch_size),
                "--num_epoch", str(args.epochs), "--text_weight", str(args.text_weight),
                "--used_col_prefixes", ",".join(args.prefixes), "--logdir", str(args.output.resolve())]
    cfg = runner.parse_args()
    runner.set_seed(args.seed)
    device = torch.device(args.device)
    model = getattr(runner, args.model)(cfg).to(device)
    text = runner.TextModel(cfg.text_hidden_size, cfg.pred_len, cfg.text_num_layers).to(device)
    datasets, loaders = {}, {}
    for split in ("train", "val", "test"):
        ds, _ = runner.get_dataset_dataloader(cfg.seq_len, cfg.label_len, cfg.pred_len,
                  split, cfg.used_col_prefixes, cfg.data_path, cfg.root_path, cfg.batch_size)
        datasets[split] = ds
    if args.scaler == "train-only":
        scaler = datasets["train"].scaler
        raw = scaler.inverse_transform(datasets["train"].data_price_x)
        from sklearn.preprocessing import StandardScaler
        fitted = StandardScaler().fit(raw)
        for ds in datasets.values():
            ds.data_price_x = fitted.transform(ds.scaler.inverse_transform(ds.data_price_x))
            ds.data_price_y = ds.data_price_x
            ds.scaler = fitted
    for split, ds in datasets.items():
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=split == "train", num_workers=0,
                            drop_last=split == "train")
        if args.smoke_batches:
            from itertools import islice
            loader = list(islice(loader, args.smoke_batches))
        loaders[split] = loader
    optimizer = torch.optim.Adam(list(model.parameters()) + list(text.parameters()), lr=cfg.lr)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        train, _ = runner._run_epoch(loaders["train"], datasets["train"], model, text, cfg, device, optimizer)
        with torch.no_grad():
            val, _ = runner._run_epoch(loaders["val"], datasets["val"], model, text, cfg, device)
        if not np.isfinite([train, val]).all():
            raise ValueError("Nonfinite loss")
        history.append({"epoch": epoch, "train_mse": train, "val_mse": val})
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best, stale = val, 0
            torch.save({"model": model.state_dict(), "text_model": text.state_dict(), "epoch": epoch,
                        "args": vars(cfg), "scaler_mean": datasets["train"].scaler.mean_.tolist(),
                        "scaler_scale": datasets["train"].scaler.scale_.tolist()}, args.output / "checkpoint.pt")
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    saved = torch.load(args.output / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    text.load_state_dict(saved["text_model"])
    with torch.no_grad():
        mse, mae, inputs, predictions, targets = runner._run_epoch(loaders["test"], datasets["test"],
                       model, text, cfg, device, collect_io=True)
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite test predictions or targets")
    baseline = inputs[:, -1:, :].expand_as(targets)
    np.savez_compressed(args.output / "predictions.npz", predictions=predictions.numpy(),
                        targets=targets.numpy(), persistence=baseline.numpy())
    metrics = {"scaled_mse": mse, "scaled_mae": mae, "persistence_scaled_mse":
               ((baseline - targets) ** 2).mean().item(), "test_windows": len(targets),
               "selected_epoch": saved["epoch"], "best_val_mse": best, "smoke_only": bool(args.smoke_batches)}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "SUCCESS.json").write_text(json.dumps({"checkpoint_sha256": sha256(args.output / "checkpoint.pt"),
                                                        "metrics": metrics}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
