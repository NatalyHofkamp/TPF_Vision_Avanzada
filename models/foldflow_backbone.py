"""Checkpoint-compatible conditioning wrapper around official FoldFlow."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .conditioning import FoldFlowConditionEncoder
from .load_foldflow import DEFAULT_CHECKPOINT, load_pretrained_foldflow

CFG_VECTORFIELD_KEYS = ("rot_vectorfield", "trans_vectorfield")


class FoldFlowBackbone(nn.Module):
    """Wrap an official FoldFlow-1 model without replacing its architecture."""

    def __init__(
        self,
        official_model: nn.Module,
        flow_matcher: Any,
        official_config: Any,
        condition_encoder: FoldFlowConditionEncoder | None = None,
    ):
        super().__init__()
        self.official_model = official_model
        self.flow_matcher = flow_matcher
        self.official_config = official_config
        node_dim = int(official_config.model.node_embed_size)
        self.condition_encoder = condition_encoder or FoldFlowConditionEncoder(
            esm_input_dim=1280, node_dim=node_dim
        )
        self._active_node_condition: torch.Tensor | None = None
        self._backbone_frozen = not any(
            parameter.requires_grad for parameter in self.official_model.parameters()
        )
        self._embedding_hook = self.official_model.embedding_layer.register_forward_hook(
            self._inject_node_condition
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: str | torch.device | None = None,
        source_root: Path | str | None = None,
        freeze_backbone: bool = True,
    ) -> "FoldFlowBackbone":
        model, flow_matcher, conf = load_pretrained_foldflow(
            checkpoint_path=checkpoint_path,
            device=device,
            source_root=source_root,
        )
        wrapper = cls(model, flow_matcher, conf)
        if freeze_backbone:
            wrapper.freeze_official_backbone()
        return wrapper.to(next(model.parameters()).device)

    def _inject_node_condition(self, _module, _inputs, output):
        if self._active_node_condition is None:
            return output
        node_embedding, edge_embedding = output
        condition = self._active_node_condition.to(
            device=node_embedding.device, dtype=node_embedding.dtype
        )
        if condition.shape != node_embedding.shape:
            raise ValueError(
                f"Condition shape {tuple(condition.shape)} does not match official "
                f"FoldFlow node embedding {tuple(node_embedding.shape)}"
            )
        return node_embedding + condition, edge_embedding

    @contextmanager
    def _conditioning_context(self, node_condition: torch.Tensor):
        if self._active_node_condition is not None:
            raise RuntimeError("Nested FoldFlow conditioning calls are not supported")
        self._active_node_condition = node_condition
        try:
            yield
        finally:
            self._active_node_condition = None

    def forward(
        self,
        foldflow_features: Mapping[str, torch.Tensor],
        conditioning: Mapping[str, torch.Tensor],
        drop_target_state: bool | torch.Tensor = False,
    ) -> dict[str, torch.Tensor]:
        residue_mask = foldflow_features["res_mask"]
        node_condition = self.condition_encoder(
            conditioning,
            residue_mask=residue_mask,
            drop_target_state=drop_target_state,
        )
        with self._conditioning_context(node_condition):
            # Do not wrap the official backbone in no_grad during training:
            # gradients must flow through the frozen backbone ops to the
            # trainable conditioning adapter. Parameters remain frozen via
            # requires_grad=False, but autograd still traces the path to the
            # adapter.
            return self.official_model(dict(foldflow_features))

    def guided_forward(
        self,
        foldflow_features: Mapping[str, torch.Tensor],
        conditioning: Mapping[str, torch.Tensor],
        guidance_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Apply translation CFG while retaining source and identity conditions."""
        conditional = self.forward(
            foldflow_features, conditioning, drop_target_state=False
        )
        unconditional = self.forward(
            foldflow_features, conditioning, drop_target_state=True
        )
        guided = dict(conditional)
        for key in CFG_VECTORFIELD_KEYS:
            guided[key] = unconditional[key] + guidance_scale * (
                conditional[key] - unconditional[key]
            )
        guided["conditional"] = conditional
        guided["unconditional"] = unconditional
        return guided

    @torch.no_grad()
    def sample_translation(
        self,
        foldflow_features: Mapping[str, torch.Tensor],
        conditioning: Mapping[str, torch.Tensor],
        guidance_scale: float = 1.0,
        num_steps: int = 50,
        min_t: float = 0.01,
        noise_scale: float = 0.0,
        center: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Integrate guided official FoldFlow vector fields from source frames."""
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        from openfold.utils import rigid_utils as ru

        features = {key: value.clone() for key, value in foldflow_features.items()}
        device = features["rigids_t"].device
        reverse_steps = torch.linspace(1.0, min_t, num_steps, device=device)
        trajectory = [features["rigids_t"].detach().cpu()]
        final_output = None
        for step_index, time_value in enumerate(reverse_steps):
            features["t"] = torch.full(
                (features["rigids_t"].shape[0],),
                float(time_value),
                device=device,
            )
            final_output = self.guided_forward(
                features, conditioning, guidance_scale=guidance_scale
            )
            if step_index == len(reverse_steps) - 1:
                break
            dt = float(time_value - reverse_steps[step_index + 1])
            flow_mask = (
                (1.0 - features["fixed_mask"]) * features["res_mask"]
            ).detach().cpu().numpy()
            _, _, next_rigids = self.flow_matcher.reverse(
                rigid_t=ru.Rigid.from_tensor_7(features["rigids_t"]),
                rot_vectorfield=final_output["rot_vectorfield"].detach().cpu().numpy(),
                trans_vectorfield=final_output["trans_vectorfield"]
                .detach()
                .cpu()
                .numpy(),
                flow_mask=flow_mask,
                t=float(time_value),
                dt=dt,
                center=center,
                noise_scale=noise_scale,
            )
            features["rigids_t"] = next_rigids.to_tensor_7().to(device)
            if self.official_config.model.embed.embed_self_conditioning:
                features["sc_ca_t"] = final_output["rigids"][..., 4:].detach()
            trajectory.append(features["rigids_t"].detach().cpu())
        return {
            "rigids": features["rigids_t"],
            "trajectory": torch.stack(trajectory),
            "model_output": final_output,
        }

    def freeze_official_backbone(self) -> None:
        self.official_model.requires_grad_(False)
        self.official_model.eval()
        self._backbone_frozen = True

    def unfreeze_official_backbone(self) -> None:
        self.official_model.requires_grad_(True)
        self._backbone_frozen = False

    def train(self, mode: bool = True) -> "FoldFlowBackbone":
        super().train(mode)
        if self._backbone_frozen:
            self.official_model.eval()
        return self

    def forward_is_frozen(self) -> bool:
        return self._backbone_frozen

    @property
    def official_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.official_model.parameters())

    @property
    def conditioning_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.condition_encoder.parameters())

    def close(self) -> None:
        self._embedding_hook.remove()
