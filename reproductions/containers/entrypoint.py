"""Profile commands. Default inventory never starts training or downloads weights."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from bootstrap import LOCK, verify

TOOLS = Path("/opt/paper_tools")


def inventory(profile):
    spec = json.loads(LOCK.read_text())["profiles"][profile]
    return {"paper": profile, "platform": "linux/amd64", "runtime": "CPU",
            "models": spec["models"], "blocked_models": spec["blocked"],
            "coverage_complete": not spec["blocked"], "training_started": False,
            "source_commit": spec.get("source_commit")}


def check(profile):
    spec = json.loads(LOCK.read_text())["profiles"][profile]
    if "source_root" in spec:
        verify(spec["source_root"], spec)
    if profile == "fnspid":
        for interpreter, imports in [("/opt/torch/bin/python", "torch,numpy,pandas,sklearn,matplotlib,tqdm"),
                                      ("/opt/keras/bin/python", "tensorflow,keras,numpy,pandas,sklearn,matplotlib,h5py")]:
            subprocess.run([interpreter, "-c", "import " + imports], check=True)
            subprocess.run([interpreter, "-m", "pip", "check"], check=True)
        for model in ("CNN", "RNN", "LSTM", "GRU"):
            folder = Path(spec["source_root"]) / "dataset_test" / f"{model}-for-Time-Series-Prediction"
            env = dict(os.environ, PYTHONPATH=str(folder))
            subprocess.run(["/opt/keras/bin/python", "-c", f"from core.{model}_modified_model import Model"],
                           cwd=folder, env=env, check=True)
        transformer = Path(spec["source_root"]) / "dataset_test/Transformer-for-Time-Series-Prediction"
        subprocess.run(["/opt/torch/bin/python", "-c", "from tst import Transformer"],
                       cwd=transformer, env=dict(os.environ, PYTHONPATH=str(transformer)), check=True)
        # Importing run.py can start author training. Parse TimesNet's definition instead.
        tree = ast.parse((Path(spec["source_root"]) / "dataset_test/TimesNet-for-Time-Series-Prediction/run.py").read_text())
        if not any(isinstance(n, ast.ClassDef) and n.name == "TimesNet" for n in tree.body):
            raise ValueError("TimesNet definition absent")
    else:
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        if profile == "fintexts":
            sys.path[:0] = [spec["source_root"], spec["source_root"] + "/forecasting_task"]
            import forecasting_task.run as author
            missing = [m for m in spec["models"] if not hasattr(author, m)]
        else:
            sys.path.insert(0, str(TOOLS))
            import finmultitime
            missing = [m for m in spec["models"] if m not in finmultitime.MODELS]
        if missing:
            raise ValueError(f"Missing model constructors: {missing}")
    result = inventory(profile)
    result["environment_check"] = "PASS"
    result["forward_training_acceptance"] = "NOT_RUN"
    print(json.dumps(result, indent=2))


def main():
    profile = os.environ.get("PAPER_PROFILE")
    if profile not in ("fnspid", "fintexts", "finmultitime"):
        raise ValueError("Image profile is not configured")
    args = sys.argv[1:]
    action = args[0] if args else "inventory"
    rest = args[1:]
    if action in ("inventory", "--help", "-h"):
        print(json.dumps(inventory(profile), indent=2))
        print("Commands: inventory | check | run <runner arguments>")
    elif action == "check":
        if rest:
            raise ValueError("check takes no extra arguments")
        check(profile)
    elif action == "run":
        if profile == "fnspid":
            script = Path(__file__).with_name("run_fnspid.py")
        elif profile == "fintexts":
            script = TOOLS / "train_fintexts.py"
            if "--author" not in rest:
                rest = ["--author", "/opt/FinTexTS", *rest]
        else:
            script = TOOLS / "finmultitime.py"
        os.execv(sys.executable, [sys.executable, str(script), *rest])
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
