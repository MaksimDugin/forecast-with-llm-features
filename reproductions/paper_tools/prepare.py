"""Acquire pinned published inputs. No training, credentials or global installs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import urllib.parse
import urllib.request

LOCK = Path(__file__).with_name("inputs.lock.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, item):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError(f"Missing/truncated input: {path}; expected {item['bytes']} bytes")
    if sha256(path) != item["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("..", ".") for p in path.parts):
        raise ValueError(f"Unsafe relative path: {value}")
    if "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*path.parts)


def acquire(spec, item, destination, download=False):
    path = Path(destination) / relative_path(item["path"])
    if path.exists():
        verify(path, item)  # Never overwrite an existing corrupt or unrelated file.
        return path
    if not download:
        raise ValueError(f"Missing input: {path}; use download explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/"
           f"{spec['revision']}/{urllib.parse.quote(item['path'], safe='/')}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            with urllib.request.urlopen(url, timeout=90) as response:
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > item["bytes"]:
                        raise ValueError(f"Unexpected download size: {item['path']}")
                    output.write(chunk)
        verify(temporary, item)
        # Hard-link creation is exclusive: a concurrent file is never replaced.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def git(*args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stdout.strip()


def verify_source(path, spec):
    path = Path(path).resolve()
    if not (path / ".git").exists():
        raise ValueError(f"Not an author checkout: {path}")
    if git("rev-parse", "HEAD", cwd=path) != spec["author_commit"]:
        raise ValueError("Author commit differs from lock")
    if git("status", "--porcelain", "--untracked-files=all", cwd=path):
        raise ValueError("Author checkout contains modifications or untracked files")
    return path


def prepare_source(destination, spec):
    destination = Path(destination).resolve()
    if destination.exists():
        return verify_source(destination, spec)
    destination.mkdir(parents=True)
    git("init", cwd=destination)
    git("remote", "add", "origin", spec["author_repository"], cwd=destination)
    git("fetch", "--depth=1", "origin", spec["author_commit"], cwd=destination)
    git("checkout", "--detach", "FETCH_HEAD", cwd=destination)
    return verify_source(destination, spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "verify", "download", "source"])
    parser.add_argument("paper", choices=["fintexts", "finmultitime"])
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    spec = json.loads(LOCK.read_text())[args.paper]
    if args.action == "list":
        print(json.dumps(spec, indent=2))
        return
    if args.destination is None:
        parser.error("--destination is required")
    if args.action == "source":
        if not spec.get("author_repository"):
            parser.error("No verified author training repository for this paper")
        print("AUTHOR_CHECKOUT =", prepare_source(args.destination, spec))
    else:
        for item in spec["files"]:
            path = acquire(spec, item, args.destination, args.action == "download")
            print("VERIFIED =", path)
    print("TRAINING_STARTED = NO")


if __name__ == "__main__":
    main()
