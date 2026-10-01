"""Fetch pinned author code only. No data, checkpoints or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

LOCK = Path(__file__).with_name("profiles.lock.json")


def command(args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stdout.strip()


def verify(path, spec):
    path = Path(path)
    if command(["rev-parse", "HEAD"], path) != spec["source_commit"]:
        raise ValueError("Author commit differs from lock")
    if command(["status", "--porcelain", "--untracked-files=all"], path):
        raise ValueError("Author source is modified")
    for item in spec.get("files", []):
        content = (path / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != item["git_blob_sha"]:
            raise ValueError(f"Source blob mismatch: {item['path']}")


def acquire(profile, destination):
    spec = json.loads(LOCK.read_text())["profiles"][profile]
    destination = Path(destination)
    if destination.exists():
        verify(destination, spec)
        return
    destination.mkdir(parents=True)
    command(["init"], destination)
    command(["remote", "add", "origin", spec["source_repository"]], destination)
    if spec.get("files"):
        command(["sparse-checkout", "init", "--no-cone"], destination)
        patterns = "\n".join('/' + i["path"] for i in spec["files"]) + "\n"
        result = subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
                                cwd=destination, input=patterns, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
    command(["fetch", "--filter=blob:none", "--depth=1", "origin", spec["source_commit"]], destination)
    command(["checkout", "--detach", "FETCH_HEAD"], destination)
    verify(destination, spec)
    print("AUTHOR_SOURCE=PASS", profile)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=["fnspid", "fintexts"])
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    acquire(args.profile, args.destination)
