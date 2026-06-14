"""Training-readiness hooks without implementing a training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import torch


class MetricsHook(Protocol):
    def __call__(self, metrics: Mapping[str, float], step: int) -> None: ...


class LoggingHook(Protocol):
    def __call__(self, message: str, **context: Any) -> None: ...


class NullMetricsHook:
    def __call__(self, metrics: Mapping[str, float], step: int) -> None:
        return None


class NullLoggingHook:
    def __call__(self, message: str, **context: Any) -> None:
        return None


class ConditioningCheckpointManager:
    """Save/load local adapters separately from immutable official weights."""

    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        filename: str,
        conditioning_module: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        **metadata: Any,
    ) -> Path:
        path = self.checkpoint_dir / filename
        payload = {
            "conditioning": conditioning_module.state_dict(),
            "metadata": metadata,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
        return path

    def load(
        self,
        path: Path | str,
        conditioning_module: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        conditioning_module.load_state_dict(payload["conditioning"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload.get("metadata", {})
