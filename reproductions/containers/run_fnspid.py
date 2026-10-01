"""Run an unchanged author script in an isolated output directory; dry-run by default."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from bootstrap import LOCK, verify


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_csvs(script):
    tree = ast.parse(Path(script).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        if not any(isinstance(target, ast.Name) and target.id in {"names_5", "names_25", "names_50"}
                   for target in node.targets):
            continue
        for item in node.value.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                raise ValueError("Author stock list is not a literal string list")
            if not item.value.lower().endswith(".csv") or "/" in item.value or "\\" in item.value:
                raise ValueError("Unexpected author stock filename")
            names.add(item.value)
    names = sorted(names)
    if not names:
        raise ValueError("No author stock filenames found")
    return names


def match_inputs(directory, names):
    lookup = {}
    for path in Path(directory).glob("*.csv"):
        key = path.name.casefold()
        if key in lookup:
            raise ValueError(f"Ambiguous case-insensitive CSV name: {path.name}")
        lookup[key] = path
    result = []
    for name in names:
        path = lookup.get(name.casefold())
        if path is None:
            raise ValueError(f"Missing author input: {name}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            required = {"Volume", "Open", "High", "Low", "Close", "Scaled_sentiment"}
            if not required.issubset(columns):
                raise ValueError(f"Missing columns in {path.name}: {sorted(required-set(columns))}")
            rows = sum(1 for _ in reader)
            if rows < 354:
                raise ValueError(f"Too few rows for author 85/15 split + history 50: {path.name}")
        result.append({"author_name": name, "input": str(path.resolve()), "sha256": digest(path), "rows": rows})
    return result


def main():
    spec = json.loads(LOCK.read_text())["profiles"]["fnspid"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=spec["models"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    source = Path(spec["source_root"])
    verify(source, spec)
    folder = source / "dataset_test" / f"{args.model}-for-Time-Series-Prediction"
    if args.output.exists():
        raise ValueError("Output already exists; no resume/overwrite")
    out = args.output.resolve()
    if out.is_relative_to(source.resolve()) or out.is_relative_to(args.data_dir.resolve()):
        raise ValueError("Output must be separate from source and input data")
    inputs = match_inputs(args.data_dir, declared_csvs(folder / "run.py"))
    interpreter = "/opt/keras/bin/python" if args.model in ("CNN", "RNN", "LSTM", "GRU") else "/opt/torch/bin/python"
    plan = {"protocol": "unchanged-author-entrypoint", "model": args.model,
            "author_commit": spec["source_commit"], "inputs": inputs,
            "command": [interpreter, "-u", "run.py"], "execute_requested": args.execute,
            "limitations": ["Runs all stocks/cases hardcoded by the selected author script",
                            "Author epoch counts and output behavior unchanged; can be lengthy",
                            "Image contains code, not published pretrained checkpoints",
                            "Exit zero is not paper metric acceptance; inspect produced outputs separately"]}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(plan, indent=2))
    workspace = out / "author_workspace"
    shutil.copytree(folder, workspace)
    (workspace / "data").mkdir(exist_ok=False)
    for item in inputs:
        target = workspace / "data" / item["author_name"]
        shutil.copyfile(item["input"], target)
        if digest(target) != item["sha256"]:
            raise ValueError("Copied input hash mismatch")
    with (out / "console.log").open("w") as log:
        result = subprocess.run(plan["command"], cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
    (out / "process_status.json").write_text(json.dumps({"returncode": result.returncode,
                        "training_requested": True, "paper_metrics_accepted": False}, indent=2))
    result.check_returncode()
    print("AUTHOR_PROCESS_EXIT=0; PAPER_METRICS_ACCEPTED=NO")


if __name__ == "__main__":
    main()
