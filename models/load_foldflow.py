"""Official FoldFlow source and checkpoint loading utilities."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

OFFICIAL_REPOSITORY = "https://github.com/DreamFold/FoldFlow"
OFFICIAL_REVISION = "9d2c260813da3c9a2bc944973953f63ea6d71203"
DEFAULT_SOURCE_ROOT = Path("third_party/foldflow_official")
DEFAULT_CHECKPOINT = Path("checkpoints/foldflow/foldflow-sfm.pth")


def configure_foldflow_import(source_root: Path | str | None = None) -> Path:
    """Add the installed official FoldFlow source tree to ``sys.path``."""
    os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
    root = Path(
        source_root
        or os.environ.get("FOLDFLOW_ROOT", DEFAULT_SOURCE_ROOT)
    ).expanduser()
    if not (root / "foldflow").is_dir() or not (root / "openfold").is_dir():
        raise FileNotFoundError(
            f"Official FoldFlow source not found at {root}. "
            "Run scripts/install_foldflow.sh first or set FOLDFLOW_ROOT."
        )
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def import_official_foldflow(source_root: Path | str | None = None):
    """Import and return the official upstream package."""
    configure_foldflow_import(source_root)
    module = importlib.import_module("foldflow")
    module_file = Path(module.__file__).resolve()
    expected_root = Path(
        source_root or os.environ.get("FOLDFLOW_ROOT", DEFAULT_SOURCE_ROOT)
    ).resolve()
    if expected_root not in module_file.parents:
        raise ImportError(
            f"Imported foldflow from unexpected location {module_file}; "
            f"expected source under {expected_root}"
        )
    return module


def _torch_load_checkpoint(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _coerce_numeric_strings(value):
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_pretrained_foldflow(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
    source_root: Path | str | None = None,
    strict: bool = True,
) -> tuple[torch.nn.Module, Any, Any]:
    """Instantiate official FoldFlow-1 and load an official release checkpoint."""
    source_path = configure_foldflow_import(source_root)
    import_official_foldflow(source_path)
    from foldflow.models import se3_fm
    from foldflow.models.components import network

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"FoldFlow checkpoint not found: {checkpoint_path}. "
            "Run scripts/download_foldflow.sh."
        )
    map_location = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError(f"Unsupported FoldFlow checkpoint schema in {checkpoint_path}")
    if "conf" in checkpoint:
        conf = checkpoint["conf"]
    else:
        model_config = source_path / "runner/config/model/benchmark.yaml"
        flow_config = source_path / "runner/config/flow_matcher/default.yaml"
        if not model_config.is_file() or not flow_config.is_file():
            raise FileNotFoundError(
                "Official FoldFlow release checkpoints omit configuration; "
                f"expected upstream configs at {model_config} and {flow_config}"
            )
        model_values = _coerce_numeric_strings(
            yaml.safe_load(model_config.read_text(encoding="utf-8"))
        )
        flow_values = _coerce_numeric_strings(
            yaml.safe_load(flow_config.read_text(encoding="utf-8"))
        )
        model_values["ipa"]["c_s"] = model_values["node_embed_size"]
        model_values["ipa"]["c_z"] = model_values["edge_embed_size"]
        model_values["ipa"]["coordinate_scaling"] = flow_values["r3"][
            "coordinate_scaling"
        ]
        model_values["axis_angle"] = flow_values["so3"]["axis_angle"]
        conf = _to_namespace(
            {"model": model_values, "flow_matcher": flow_values}
        )
    model_conf = conf.model
    flow_conf = getattr(conf, "flow_matcher", None)
    if flow_conf is None:
        flow_conf = getattr(conf, "diffuser", None)
    if flow_conf is None:
        raise ValueError("Checkpoint does not contain FoldFlow flow-matcher config")

    flow_matcher = se3_fm.SE3FlowMatcher(flow_conf)
    model = network.VectorFieldNetwork(model_conf, flow_matcher)
    state_dict = {
        key.replace("module.", "").replace("score_model.", "vectorfield."): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict, strict=strict)
    model.to(map_location)
    model.eval()
    return model, flow_matcher, conf


def load_model(model_path=None, **kwargs):
    """Backward-compatible alias for the official pretrained loader."""
    model, _, _ = load_pretrained_foldflow(
        checkpoint_path=model_path or DEFAULT_CHECKPOINT, **kwargs
    )
    return model
