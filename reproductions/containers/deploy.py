"""VS Code paper workflows: local build/check and verified image transfer over SSH."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import manage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXAMPLE = HERE / "deploy.example.json"
DEFAULT_CONFIG = HERE / "deploy.local.json"


def load_config(path):
    config = json.loads((path if path.exists() else EXAMPLE).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported deployment config schema")
    if not isinstance(config.get("local_cgroups_disabled"), bool):
        raise ValueError("local_cgroups_disabled must be boolean")
    return config


def ssh_target(config):
    target = config.get("ssh_target")
    if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", target):
        raise ValueError("Set ssh_target in deploy.local.json to your SSH alias or user@host; no server was contacted")
    return target


def resolve_program(name):
    if not isinstance(name, str) or not name:
        raise ValueError("Executable name/path is missing")
    resolved = shutil.which(name)
    if resolved:
        return resolved
    path = Path(name)
    if path.is_file():
        return str(path.resolve())
    if os.name == "nt":
        if name == "podman":
            candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Podman/podman.exe"
        elif name == "ssh":
            candidate = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/OpenSSH/ssh.exe"
        else:
            candidate = None
        if candidate and candidate.is_file():
            return str(candidate)
    raise ValueError(f"Executable not found: {name}")


def engine(config):
    result = [resolve_program(config["podman"])]
    connection = config.get("connection")
    if connection:
        if not isinstance(connection, str):
            raise ValueError("Podman connection must be a string or null")
        result.extend(["--connection", connection])
    return result


def ssh_prefix(config):
    return [resolve_program(config["ssh"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ssh_target(config)]


def remote(config, args):
    # OpenSSH joins remote arguments into a shell command: quote each token here.
    return [*ssh_prefix(config), shlex.join(["podman", *args])]


def image_ref(profile, revision):
    if profile not in manage.PROFILES or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid profile or source revision")
    return f"localhost/{profile}:cpu-{revision[:12]}"


def clean_revision():
    revision = manage.git("rev-parse", "HEAD")
    if manage.git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Use a clean pinned worktree; current checkout contains WIP")
    return revision


def cache_dir(config):
    configured = config.get("local_output_dir")
    cache = Path(configured).expanduser().resolve() if configured else ROOT.parent / f"{ROOT.name}.paper-ops"
    if cache.resolve().is_relative_to(ROOT):
        raise ValueError("local_output_dir must be outside this checkout")
    return cache


def read_log(path, lines=40):
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        from collections import deque
        return "".join(deque(stream, maxlen=lines))


def call(command, log, *, check=True):
    with log.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if check and result.returncode:
        raise RuntimeError(f"Exit {result.returncode}; log={log}\n{read_log(log)}")
    return result.returncode


def inspect_local(config, ref, revision, log):
    call([*engine(config), "image", "inspect", ref], log)
    image = json.loads(log.read_text(encoding="utf-8"))[0]
    labels = image.get("Labels") or image.get("Config", {}).get("Labels") or {}
    if labels.get("org.opencontainers.image.revision") != revision:
        raise ValueError(f"Image source revision mismatch: {ref}")
    if image.get("Architecture") != "amd64" or image.get("Os") != "linux":
        raise ValueError(f"Unexpected image platform: {ref}")
    value = image["Id"].removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Malformed image ID")
    return value


def remote_id(config, ref, log):
    call(remote(config, ["image", "inspect", "--format", "{{.Id}}", ref]), log)
    value = log.read_text(encoding="utf-8").strip().removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Malformed remote image ID")
    return value


def transfer(config, ref, folder):
    """Binary pipeline; separately check both exits and always reap owned subprocesses."""
    with (folder / "save.stderr.log").open("wb") as save_err, \
         (folder / "load.stdout.log").open("wb") as load_out, \
         (folder / "load.stderr.log").open("wb") as load_err:
        saver = subprocess.Popen([*engine(config), "save", "--format", "oci-archive", ref],
                                 stdout=subprocess.PIPE, stderr=save_err)
        loader = None
        try:
            loader = subprocess.Popen(remote(config, ["load"]), stdin=saver.stdout,
                                      stdout=load_out, stderr=load_err)
            saver.stdout.close()
            load_rc = loader.wait()
            save_rc = saver.wait()
            if save_rc or load_rc:
                raise RuntimeError(f"Image transfer failed: save={save_rc}, load={load_rc}; logs={folder}")
        finally:
            if saver.stdout and not saver.stdout.closed:
                saver.stdout.close()
            for process in (loader, saver):
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()


def deploy_images(config, profiles, revision, output, manifest, save, *, transfer_images):
    # Verify all local images and the remote engine before the first image load.
    ssh_target(config)
    expected = {}
    for profile in profiles:
        folder = output / profile
        folder.mkdir()
        expected[profile] = inspect_local(config, image_ref(profile, revision), revision, folder / "local-image.json")
    call(remote(config, ["info", "--format", "json"]), output / "remote-engine.json")
    host = json.loads((output / "remote-engine.json").read_text(encoding="utf-8"))["host"]
    if host.get("arch") != "amd64" or host.get("os") != "linux":
        raise ValueError("Remote engine must be linux/amd64 for these CPU images")
    for profile in profiles:
        ref = image_ref(profile, revision)
        folder = output / profile
        item = {"tag": ref, "local_image_id": expected[profile]}
        manifest["images"][profile] = item
        save()
        if transfer_images:
            exists = call(remote(config, ["image", "exists", ref]), folder / "remote-exists.log", check=False)
            if exists == 0:
                current = remote_id(config, ref, folder / "remote-before.id")
                if current != expected[profile]:
                    raise ValueError(f"Remote tag already points to a different image: {ref}; refusing to replace it")
                item["transfer"] = "SKIPPED_IDENTICAL_IMAGE"
            elif exists == 1:
                print(f"TRANSFER={profile}", flush=True)
                transfer(config, ref, folder)
                item["transfer"] = "PASS"
            else:
                raise ValueError(f"Remote image exists returned unexpected status {exists}")
        actual = remote_id(config, ref, folder / "remote-after.id")
        item["remote_image_id"] = actual
        if actual != expected[profile]:
            raise ValueError(f"Local/remote image ID mismatch: {profile}")
        # Run by immutable ID; no volumes, ports, training, or WSL workaround on the server.
        call(remote(config, ["run", "--rm", "--network=none", "sha256:" + actual, "check"]), folder / "environment-check.log")
        item["environment_check"] = "PASS"
        save()
        print(f"{profile}: IMAGE_ID_MATCH=PASS ENVIRONMENT_CHECK=PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init-config", "plan", "build", "check", "deploy", "server-check", "logs"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("all", *manage.PROFILES), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "init-config":
        if args.config.exists():
            print(f"CONFIG_PRESERVED={args.config}")
        else:
            args.config.parent.mkdir(parents=True, exist_ok=True)
            with args.config.open("x", encoding="utf-8") as stream:
                stream.write(EXAMPLE.read_text(encoding="utf-8"))
            assert load_config(args.config) == config
            print(f"CONFIG_CREATED={args.config}")
        return
    profiles = list(manage.PROFILES) if args.profile == "all" else [args.profile]
    if args.action == "plan":
        revision = manage.git("rev-parse", "HEAD")
        print(json.dumps({"images": {p: image_ref(p, revision) for p in profiles},
                          "ssh_configured": bool(config.get("ssh_target")),
                          "remote_data_dir": config.get("remote_data_dir"),
                          "remote_runs_dir": config.get("remote_runs_dir"),
                          "remote_cache_dir": config.get("remote_cache_dir"),
                          "deploy_scope": "image transfer + environment checks",
                          "training_started": False}, indent=2))
        return
    cache = cache_dir(config)
    if args.action == "logs":
        latest = json.loads((cache / "latest.json").read_text(encoding="utf-8"))
        folder = Path(latest["output"])
        for log in sorted(folder.rglob("*.log")):
            if len(log.relative_to(folder).parts) > 2 or log.parent.name == "context":
                continue
            print(f"LOG={log}\n{read_log(log, 15)}")
        return
    revision = clean_revision()
    if args.action in ("deploy", "server-check"):
        ssh_target(config)  # Fail before cache mutation or local image inspection when server is unknown.
    cache.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = cache / f"{args.action}-{stamp}"
    output.mkdir()
    manifest = {"action": args.action, "revision": revision, "profiles": profiles,
                "status": "RUNNING", "images": {}, "training_started": False,
                "server_model_acceptance": "NOT_RUN"}
    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    (cache / "latest.json").write_text(json.dumps({"output": str(output)}, indent=2), encoding="utf-8")
    try:
        if args.action in ("build", "check"):
            command = [sys.executable, str(HERE / "manage.py"), args.action,
                       "--output", str(output / "result"), "--podman", resolve_program(config["podman"]),
                       "--profiles", *profiles]
            if config.get("connection"):
                command += ["--connection", config["connection"]]
            if args.action == "check" and config["local_cgroups_disabled"]:
                command.append("--local-cgroups-disabled")
            print(f"LOCAL_{args.action.upper()}={','.join(profiles)}; LOGS={output}", flush=True)
            call(command, output / "workflow.log")
        else:
            deploy_images(config, profiles, revision, output, manifest, save, transfer_images=args.action == "deploy")
        manifest["status"] = "PASS"
        save()
        print(f"{args.action.upper()}=PASS; MANIFEST={output / 'manifest.json'}")
        print("TRAINING_STARTED=NO; SERVER_MODEL_ACCEPTANCE=NOT_RUN")
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STOP={error}", file=sys.stderr)
        sys.exit(1)
