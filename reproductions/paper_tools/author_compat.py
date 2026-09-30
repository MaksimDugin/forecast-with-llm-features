"""Opt-in device correction: executes a narrow copy, never edits author files."""
import importlib
import inspect
import textwrap


def portable_method(method):
    try:
        source = textwrap.dedent(inspect.getsource(method))
    except (TypeError, OSError) as exc:
        raise ValueError("Cannot verify method source") from exc
    if source.count(".cuda()") != 1 or "values" not in inspect.signature(method).parameters:
        raise ValueError("Unexpected method: refuse compatibility replacement")
    source = source.replace(".cuda()", ".to(values.device)")
    scope = dict(method.__globals__)
    exec(compile(source, "<opt-in-author-device-fix>", "exec"), scope)
    return scope[method.__name__]


def apply_device_fix():
    module = importlib.import_module("layers.AutoCorrelation")
    for name in ("time_delay_agg_inference", "time_delay_agg_full"):
        method = getattr(module.AutoCorrelation, name)
        setattr(module.AutoCorrelation, name, portable_method(method))
    return "AutoCorrelation .cuda() -> .to(values.device), in-memory only"
