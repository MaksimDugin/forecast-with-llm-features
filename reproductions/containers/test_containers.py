"""Packaging contracts; no torch/TensorFlow import or training."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap  # noqa: E402 - sibling scripts are intentionally loaded from this directory
import entrypoint  # noqa: E402 - sibling scripts are intentionally loaded from this directory
import manage  # noqa: E402 - sibling scripts are intentionally loaded from this directory
import run_fnspid  # noqa: E402 - sibling scripts are intentionally loaded from this directory


def test_three_profiles_and_honest_coverage():
    lock = json.loads(bootstrap.LOCK.read_text())
    assert list(lock["profiles"]) == list(manage.PROFILES)
    assert lock["platform"] == "linux/amd64"
    assert "@sha256:" in lock["base_image"]
    assert [len(lock["profiles"][p]["models"]) for p in manage.PROFILES] == [6, 12, 5]
    assert len(lock["profiles"]["finmultitime"]["blocked"]) == 3
    assert not entrypoint.inventory("finmultitime")["coverage_complete"]
    for p in manage.PROFILES:
        assert not entrypoint.inventory(p)["training_started"]


@pytest.mark.parametrize("profile", manage.PROFILES)
def test_container_default_is_inventory_and_source_pinned(profile):
    content = (HERE / f"Containerfile.{profile}").read_text()
    assert content.startswith("FROM " + json.loads(bootstrap.LOCK.read_text())["base_image"] + "\n")
    assert 'CMD ["inventory"]' in content
    assert "--execute" not in content
    assert "SOURCE_REVISION" in content
    assert "pip check" in content
    assert "os-packages.txt" in content


def test_keras_torch_isolation():
    content = (HERE / "Containerfile.fnspid").read_text()
    assert "-m venv /opt/torch" in content and "-m venv /opt/keras" in content
    assert "/opt/keras/bin/python -m pip install" in content
    assert "torch==2.5.0" in content


def test_server_runtime_has_no_cgroup_workaround_by_default():
    command = manage.runtime_command(["podman"], "image")
    assert "--cgroups=disabled" not in command
    assert "--network=none" in command and command[-1] == "check"
    assert "--cgroups=disabled" in manage.runtime_command(["podman"], "image", True)


def test_source_verifier_rejects_wrong_commit_and_wip(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    bootstrap.verify(tmp_path, {"source_commit": commit})
    with pytest.raises(ValueError, match="commit differs"):
        bootstrap.verify(tmp_path, {"source_commit": "0" * 40})
    (tmp_path / "wip.txt").write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        bootstrap.verify(tmp_path, {"source_commit": commit})


def test_case_matching_and_required_columns(tmp_path):
    cols = "Volume,Open,High,Low,Close,Scaled_sentiment\n"
    (tmp_path / "ko.csv").write_text(cols + "1,2,3,4,5,0\n" * 360)
    inputs = run_fnspid.match_inputs(tmp_path, ["KO.csv"])
    assert inputs[0]["rows"] == 360 and inputs[0]["author_name"] == "KO.csv"
    with pytest.raises(ValueError, match="Missing author input"):
        run_fnspid.match_inputs(tmp_path, ["AMD.csv"])
    (tmp_path / "bad.csv").write_text("Close\n1\n")
    with pytest.raises(ValueError, match="Missing columns"):
        run_fnspid.match_inputs(tmp_path, ["bad.csv"])
    (tmp_path / "KO.csv").write_text(cols)
    with pytest.raises(ValueError, match="Ambiguous"):
        run_fnspid.match_inputs(tmp_path, ["KO.csv"])


def test_context_contains_only_versioned_inputs(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    own = root / "reproductions/containers"
    tools = root / "reproductions/paper_tools"
    own.mkdir(parents=True)
    tools.mkdir()
    (own / "manage.py").write_text("committed")
    (own / ".containerignore").write_text("*\n")
    (tools / "train.py").write_text("committed")
    (root / "secret.env").write_text("not part of context")
    paths = ["reproductions/containers/manage.py", "reproductions/containers/.containerignore", "reproductions/paper_tools/train.py"]
    def fake_git(*args):
        if args[0] == "rev-parse":
            return "a" * 40
        if args[0] == "status":
            return ""
        return "\n".join(paths)
    monkeypatch.setattr(manage, "ROOT", root)
    monkeypatch.setattr(manage, "HERE", own)
    monkeypatch.setattr(manage, "git", fake_git)
    revision, hashes = manage.snapshot(tmp_path / "context")
    assert revision == "a" * 40 and set(hashes) == set(paths)
    assert not (tmp_path / "context/secret.env").exists()
    with pytest.raises(ValueError, match="already exists"):
        manage.snapshot(tmp_path / "context")


def test_snapshot_refuses_workflow_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "git", lambda *args: " M reproductions/paper_tools/train.py" if args[0] == "status" else "a" * 40)
    with pytest.raises(ValueError, match="committed and clean"):
        manage.snapshot(tmp_path / "context")
    assert not (tmp_path / "context").exists()


def test_modules_do_not_import_ml_runtime_at_module_scope():
    import ast
    for module in (bootstrap, entrypoint, manage, run_fnspid):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in imports if isinstance(node, ast.Import) for alias in node.names}
        names |= {node.module.split(".")[0] for node in imports if isinstance(node, ast.ImportFrom)}
        assert not names & {"torch", "tensorflow", "numpy", "pandas"}


def test_declared_csvs_excludes_output_suffixes(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("names_5 = ['KO.csv']\nnames_50 = ['KO.csv', 'AMD.csv']\nsaved = f'{symbol}_eval.csv'\nother = '.csv'\n")
    assert run_fnspid.declared_csvs(script) == ["AMD.csv", "KO.csv"]


def test_inventory_matches_runner_model_choices():
    import ast
    lock = json.loads(bootstrap.LOCK.read_text())
    tools = HERE.parent / "paper_tools"
    for profile, filename in (("fintexts", "run_fintexts.py"), ("finmultitime", "finmultitime.py")):
        tree = ast.parse((tools / filename).read_text())
        declaration = next(node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in node.targets))
        assert set(ast.literal_eval(declaration.value)) == set(lock["profiles"][profile]["models"])
