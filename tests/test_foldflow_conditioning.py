from types import SimpleNamespace

import torch
from torch import nn

from models.conditioning import (
    ESMConditionEncoder,
    KinaseConditionEncoder,
    StateConditionEncoder,
)
from models.foldflow_backbone import FoldFlowBackbone


class FakeEmbedder(nn.Module):
    def forward(self, *, seq_idx, t, fixed_mask, self_conditioning_ca):
        batch_size, length = seq_idx.shape
        return torch.zeros(batch_size, length, 8), torch.zeros(
            batch_size, length, length, 4
        )


class FakeOfficialFoldFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = FakeEmbedder()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, features):
        node, _ = self.embedding_layer(
            seq_idx=features["seq_idx"],
            t=features["t"],
            fixed_mask=features["fixed_mask"],
            self_conditioning_ca=features["sc_ca_t"],
        )
        trans = node[..., :3] * self.scale
        rot = trans[..., :, None].expand(*trans.shape, 3)
        return {
            "trans_vectorfield": trans,
            "rot_vectorfield": rot,
            "rigids": features["rigids_t"],
        }


class FakeConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, conditioning, residue_mask, drop_target_state=False):
        base = conditioning["esm_embedding"][..., :8] * self.weight
        if not drop_target_state:
            base = base + conditioning["target_state"][:, None, None]
        return base * residue_mask[..., None]


def test_condition_encoders_shapes():
    esm = ESMConditionEncoder(input_dim=16, output_dim=8)
    kinase = KinaseConditionEncoder(output_dim=8)
    state = StateConditionEncoder(output_dim=8)
    assert esm(torch.randn(2, 5, 16), torch.ones(2, 5)).shape == (2, 5, 8)
    assert kinase(torch.tensor([0, 12])).shape == (2, 8)
    assert state(torch.tensor([0, 1]), torch.tensor([False, True])).shape == (2, 8)
    assert torch.equal(
        state(torch.tensor([1]), torch.tensor([True])),
        torch.zeros(1, 8),
    )


def test_cfg_drops_only_target_state_and_uses_standard_formula():
    config = SimpleNamespace(
        model=SimpleNamespace(
            node_embed_size=8,
            embed=SimpleNamespace(embed_self_conditioning=False),
        )
    )
    model = FoldFlowBackbone(
        FakeOfficialFoldFlow(),
        flow_matcher=None,
        official_config=config,
        condition_encoder=FakeConditionEncoder(),
    )
    features = {
        "seq_idx": torch.arange(4)[None],
        "t": torch.tensor([0.5]),
        "fixed_mask": torch.zeros(1, 4),
        "res_mask": torch.ones(1, 4),
        "sc_ca_t": torch.zeros(1, 4, 3),
        "rigids_t": torch.zeros(1, 4, 7),
    }
    conditioning = {
        "esm_embedding": torch.randn(1, 4, 8),
        "kinase_id": torch.tensor([2]),
        "target_state": torch.tensor([1]),
    }
    guided = model.guided_forward(features, conditioning, guidance_scale=2.5)
    expected = guided["unconditional"]["trans_vectorfield"] + 2.5 * (
        guided["conditional"]["trans_vectorfield"]
        - guided["unconditional"]["trans_vectorfield"]
    )
    assert torch.allclose(guided["trans_vectorfield"], expected)
    assert torch.equal(features["rigids_t"], guided["conditional"]["rigids"])
    assert torch.equal(features["rigids_t"], guided["unconditional"]["rigids"])


def test_frozen_official_backbone_stays_in_eval_mode():
    config = SimpleNamespace(
        model=SimpleNamespace(
            node_embed_size=8,
            embed=SimpleNamespace(embed_self_conditioning=False),
        )
    )
    model = FoldFlowBackbone(
        FakeOfficialFoldFlow(),
        flow_matcher=None,
        official_config=config,
        condition_encoder=FakeConditionEncoder(),
    )
    model.freeze_official_backbone()
    model.train()
    assert model.condition_encoder.training
    assert not model.official_model.training
