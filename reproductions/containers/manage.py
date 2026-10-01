"""Build, check, or export three paper images. Requires a committed clean checkout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILES = ("fnspid", "fintexts", "finmultitime")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def snapshot(destination):
    """Copy only committed packaging/workflow files; reject changed or missing inputs."""
    revision = git("rev-parse", "HEAD")
    prefixes = ("reproductions/containers/", "reproductions/paper_tools/")
    status = git("status", "--porcelain", "--untracked-files=all", "--", *prefixes)
    if status:
        raise ValueError("Container/workflow paths must be committed and clean")
    paths = git("ls-files", "--", *prefixes).splitlines()
    if not paths or "reproductions/containers/manage.py" not in paths:
        raise ValueError("Packaging files are not committed; fetch the container preparation commit")
    if destination.exists():
        raise ValueError("Build context already exists")
    destination.mkdir(parents=True)
    hashes = {}
    for relative in paths:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Nonregular packaging input: {relative}")
        target = destination / Path(relative).relative_to("reproductions")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = sha256(target)
    shutil.copyfile(HERE / ".containerignore", destination / ".containerignore")
    return revision, hashes


def run(command, log):
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"Exit {result.returncode}; log {log}\n" + "\n".join(tail))


def runtime_command(engine, image, disabled=False):
    return [*engine, "run", "--rm", "--network=none", *(["--cgroups=disabled"] if disabled else []), image, "check"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "export"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--podman", default=shutil.which("podman"))
    parser.add_argument("--connection", help="Optional explicit Podman remote connection")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--local-cgroups-disabled", action="store_true", help="WSL workaround for check only; not a server default")
    args = parser.parse_args()
    if not args.podman:
        parser.error("Podman was not found; supply --podman /path/to/podman")
    if args.local_cgroups_disabled and args.action != "check":
        parser.error("--local-cgroups-disabled is allowed only for check")
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        parser.error("--output must be a new directory outside this checkout")
    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all", "--", "reproductions/containers", "reproductions/paper_tools"):
        parser.error("Container/workflow paths must be committed and clean")
    output.mkdir(parents=True)
    engine = [args.podman, *(["--connection", args.connection] if args.connection else [])]
    lock = json.loads((HERE / "profiles.lock.json").read_text())
    manifest = {"action": args.action, "source_revision": revision, "base_image": lock["base_image"],
                "platform": lock["platform"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "models_acceptance": "PENDING_SERVER_TESTS", "training_started": False,
                "local_cgroups_disabled": args.local_cgroups_disabled, "images": {}}
    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    try:
        run([*engine, "info"], output / "engine.log")
        if args.action == "build":
            revision, hashes = snapshot(output / "context")
            manifest["context_sha256"] = hashes
        for profile in dict.fromkeys(args.profiles):
            image = f"localhost/{profile}:cpu-{revision[:12]}"
            print(f"{args.action.upper()}={profile}", flush=True)
            item = {"tag": image, "models": lock["profiles"][profile]["models"],
                    "blocked_models": lock["profiles"][profile]["blocked"]}
            manifest["images"][profile] = item
            save()
            if args.action == "build":
                command = [*engine, "build", "--platform", lock["platform"], "--build-arg", f"SOURCE_REVISION={revision}",
                           "-f", str(output / "context" / "containers" / f"Containerfile.{profile}"),
                           "-t", image, str(output / "context")]
                run(command, output / f"build-{profile}.log")
            inspect = output / f"inspect-{profile}.json"
            run([*engine, "image", "inspect", image], inspect)
            metadata = json.loads(inspect.read_text())[0]
            item["image_id"] = metadata["Id"]
            labels = metadata.get("Labels") or metadata.get("Config", {}).get("Labels") or {}
            if labels.get("org.opencontainers.image.revision") != revision:
                raise ValueError(f"Image revision does not match checkout: {profile}")
            if metadata.get("Architecture") != "amd64" or metadata.get("Os") != "linux":
                raise ValueError(f"Unexpected image platform: {profile}")
            if args.action == "check":
                run(runtime_command(engine, image, args.local_cgroups_disabled), output / f"check-{profile}.log")
                item["environment_check"] = "PASS"
            elif args.action == "export":
                archive = output / f"{profile}.oci.tar"
                run([*engine, "save", "--format", "oci-archive", "--output", str(archive), image], output / f"export-{profile}.log")
                item["archive"] = archive.name
                item["archive_sha256"] = sha256(archive)
            item["status"] = "PASS"
            save()
        manifest["status"] = "PASS"
        save()
        print(f"MANIFEST={output / 'manifest.json'}")
        print("TRAINING_STARTED=NO; SERVER_MODEL_ACCEPTANCE=PENDING")
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
