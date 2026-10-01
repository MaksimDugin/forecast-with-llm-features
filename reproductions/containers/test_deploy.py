"""SSH deployment contracts without connecting to a server or starting ML jobs."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy  # noqa: E402 - CLI modules are intentionally loaded from their directory


@pytest.mark.parametrize("target", [None, "", "-oProxyCommand=bad", "host;evil", "host\nother", "user@host command"])
def test_unconfigured_or_invalid_target_is_rejected(target):
    with pytest.raises(ValueError, match="ssh_target"):
        deploy.ssh_target({"ssh_target": target})


def test_remote_command_uses_openssh_and_quoted_tokens(monkeypatch):
    monkeypatch.setattr(deploy, "resolve_program", lambda name: name)
    config = {"ssh": "ssh", "ssh_target": "research-server"}
    command = deploy.remote(config, ["run", "--rm", "image", "argument with spaces; data"])
    assert command[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "research-server"]
    assert command[6] == "podman run --rm image 'argument with spaces; data'"
    assert "--cgroups=disabled" not in command[6]


def test_plan_never_resolves_program_or_contacts_server(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["deploy.py", "plan"])
    monkeypatch.setattr(deploy.manage, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(deploy, "resolve_program", lambda *args: pytest.fail("Executable lookup in dry plan"))
    monkeypatch.setattr(deploy.subprocess, "run", lambda *args, **kw: pytest.fail("Network/process in plan"))
    deploy.main()
    result = json.loads(capsys.readouterr().out)
    assert not result["ssh_configured"] and not result["training_started"]
    assert set(result["images"]) == set(deploy.manage.PROFILES)


def test_missing_server_blocks_deploy_before_any_side_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["deploy.py", "deploy"])
    monkeypatch.setattr(deploy, "clean_revision", lambda: "a" * 40)
    monkeypatch.setattr(deploy, "cache_dir", lambda config: tmp_path / "cache")
    monkeypatch.setattr(deploy, "deploy_images", lambda *args, **kw: pytest.fail("Server invocation"))
    with pytest.raises(ValueError, match="ssh_target"):
        deploy.main()
    assert not (tmp_path / "cache").exists()


def test_local_config_creation_preserves_existing_config(tmp_path, monkeypatch, capsys):
    config = tmp_path / "deploy.local.json"
    monkeypatch.setattr(sys, "argv", ["deploy.py", "init-config", "--config", str(config)])
    deploy.main()
    initial = config.read_text()
    assert "CONFIG_CREATED" in capsys.readouterr().out
    deploy.main()
    assert "CONFIG_PRESERVED" in capsys.readouterr().out
    assert config.read_text() == initial


def test_image_metadata_must_match_revision_and_platform(tmp_path, monkeypatch):
    log = tmp_path / "inspect.json"
    monkeypatch.setattr(deploy, "call", lambda *args, **kw: 0)
    monkeypatch.setattr(deploy, "engine", lambda config: ["podman"])
    metadata = {"Id": "sha256:" + "b" * 64, "Architecture": "amd64", "Os": "linux",
                "Labels": {"org.opencontainers.image.revision": "a" * 40}}
    log.write_text(json.dumps([metadata]))
    assert deploy.inspect_local({}, "ref", "a" * 40, log) == "b" * 64
    with pytest.raises(ValueError, match="revision mismatch"):
        deploy.inspect_local({}, "ref", "c" * 40, log)
    metadata["Architecture"] = "arm64"
    log.write_text(json.dumps([metadata]))
    with pytest.raises(ValueError, match="platform"):
        deploy.inspect_local({}, "ref", "a" * 40, log)


def mock_deploy(tmp_path, monkeypatch, remote_value="b" * 64, exists=1):
    config = {"ssh": "ssh", "ssh_target": "research-server"}
    manifest = {"images": {}}
    calls = []
    monkeypatch.setattr(deploy, "inspect_local", lambda *args: "b" * 64)
    monkeypatch.setattr(deploy, "resolve_program", lambda name: name)
    monkeypatch.setattr(deploy, "remote_id", lambda *args: remote_value)
    def fake_call(command, log, **kw):
        calls.append(command[-1])
        if " info " in command[-1]:
            log.write_text(json.dumps({"host": {"arch": "amd64", "os": "linux"}}))
        return exists if "image exists" in command[-1] else 0
    monkeypatch.setattr(deploy, "call", fake_call)
    transfer = Mock()
    monkeypatch.setattr(deploy, "transfer", transfer)
    return config, manifest, calls, transfer


def test_transfer_verified_before_environment_check(tmp_path, monkeypatch):
    config, manifest, calls, transfer = mock_deploy(tmp_path, monkeypatch)
    deploy.deploy_images(config, ["fintexts"], "a" * 40, tmp_path, manifest, lambda: None, transfer_images=True)
    assert transfer.call_count == 1
    check = next(command for command in calls if " run " in command)
    assert "--network=none" in check and "sha256:" + "b" * 64 in check
    assert "--execute" not in check and "--cgroups=disabled" not in check
    assert manifest["images"]["fintexts"]["environment_check"] == "PASS"


def test_id_mismatch_prevents_any_container_start(tmp_path, monkeypatch):
    config, manifest, calls, transfer = mock_deploy(tmp_path, monkeypatch, remote_value="c" * 64)
    with pytest.raises(ValueError, match="ID mismatch"):
        deploy.deploy_images(config, ["fintexts"], "a" * 40, tmp_path, manifest, lambda: None, transfer_images=True)
    assert transfer.call_count == 1
    assert not any(" run " in command for command in calls)


def test_existing_different_tag_is_preserved(tmp_path, monkeypatch):
    config, manifest, calls, transfer = mock_deploy(tmp_path, monkeypatch, remote_value="c" * 64, exists=0)
    with pytest.raises(ValueError, match="refusing to replace"):
        deploy.deploy_images(config, ["fnspid"], "a" * 40, tmp_path, manifest, lambda: None, transfer_images=True)
    transfer.assert_not_called()
    assert not any(" run " in command for command in calls)


def test_binary_transfer_checks_both_processes_and_reaps_on_loader_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "engine", lambda config: ["podman"])
    monkeypatch.setattr(deploy, "remote", lambda config, args: ["ssh", "host", "podman load"])
    saver = Mock()
    saver.stdout.closed = False
    saver.poll.return_value = None
    monkeypatch.setattr(deploy.subprocess, "Popen", Mock(side_effect=[saver, OSError("ssh missing")]))
    with pytest.raises(OSError, match="ssh missing"):
        deploy.transfer({}, "ref", tmp_path)
    saver.stdout.close.assert_called()
    saver.terminate.assert_called_once()
    saver.wait.assert_called_once_with(timeout=10)


def test_tasks_have_no_bank_parameters_or_training():
    tasks = json.loads((HERE.parents[1] / ".vscode/tasks.json").read_text())
    assert len(tasks["tasks"]) == 7
    for task in tasks["tasks"]:
        assert task["type"] == "process"
        assert task["options"]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "deploy.py" in task["args"][0]
    content = json.dumps(tasks)
    assert "debby" not in content and "portfolio" not in content and "--execute" not in content
    example = json.loads(deploy.EXAMPLE.read_text())
    assert example["ssh_target"] is None
    for field in ("remote_data_dir", "remote_runs_dir", "remote_cache_dir"):
        assert example[field] is None
