from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from omg.cli.generation.train import _load_weights_only_checkpoint
from omg.generation.architecture import MODEL_ARCHITECTURE_KEY, build_model_architecture_contract


class _TinyConditionedModel(nn.Module):
    def __init__(self, *, text_width: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(text_width, text_width)
        self.frame_cond_injection = "per_layer_film"
        self.history_pos_encoding = "sinusoidal"
        self.backbone.self_attention_qk_norm = True
        self.backbone.cross_attention_qk_norm = True

    @property
    def denoiser(self):
        return self.backbone


def _save_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    torch.save({"state_dict": state_dict}, path)


def test_ignore_prefixes_loads_all_core_weights_and_retains_new_text_encoder(tmp_path: Path) -> None:
    source = _TinyConditionedModel(text_width=2)
    target = _TinyConditionedModel(text_width=3)
    with torch.no_grad():
        source.backbone.weight.fill_(4.0)
        source.backbone.bias.fill_(5.0)
        source.text_encoder.weight.fill_(6.0)
        target.text_encoder.weight.fill_(9.0)
        target.text_encoder.bias.fill_(8.0)
    initial_text_state = {key: value.clone() for key, value in target.text_encoder.state_dict().items()}
    checkpoint_path = tmp_path / "weights.ckpt"
    _save_checkpoint(checkpoint_path, source.state_dict())

    _load_weights_only_checkpoint(
        target,
        checkpoint_path,
        strict=False,
        ignore_prefixes=["text_encoder."],
        legacy_attention_contract="self-and-cross",
    )

    torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
    torch.testing.assert_close(target.backbone.bias, source.backbone.bias)
    for key, value in target.text_encoder.state_dict().items():
        torch.testing.assert_close(value, initial_text_state[key])


@pytest.mark.parametrize("failure", ["missing", "unexpected", "mismatched"])
def test_ignore_prefixes_requires_an_exact_non_ignored_contract_before_loading(
    tmp_path: Path,
    failure: str,
) -> None:
    source = _TinyConditionedModel()
    target = _TinyConditionedModel()
    state_dict = dict(source.state_dict())
    if failure == "missing":
        del state_dict["backbone.bias"]
    elif failure == "unexpected":
        state_dict["backbone.extra"] = torch.ones(1)
    else:
        state_dict["backbone.weight"] = torch.full((1, 2), 7.0)
    checkpoint_path = tmp_path / f"{failure}.ckpt"
    _save_checkpoint(checkpoint_path, state_dict)
    initial_backbone = {key: value.clone() for key, value in target.backbone.state_dict().items()}

    with pytest.raises(RuntimeError, match=rf"{failure}_non_ignored"):
        _load_weights_only_checkpoint(
            target,
            checkpoint_path,
            strict=False,
            ignore_prefixes=["text_encoder."],
            legacy_attention_contract="self-and-cross",
        )

    for key, value in target.backbone.state_dict().items():
        torch.testing.assert_close(value, initial_backbone[key])


def test_non_strict_loading_without_ignore_prefixes_preserves_overlap_copy_behavior(tmp_path: Path) -> None:
    target = _TinyConditionedModel()
    state_dict = dict(target.state_dict())
    state_dict["backbone.weight"] = torch.full((1, 2), 7.0)
    checkpoint_path = tmp_path / "legacy-non-strict.ckpt"
    _save_checkpoint(checkpoint_path, state_dict)

    _load_weights_only_checkpoint(target, checkpoint_path, strict=False)

    torch.testing.assert_close(target.backbone.weight[0], torch.full((2,), 7.0))


def test_ignored_subtree_requires_explicit_contract_for_legacy_checkpoint(tmp_path: Path) -> None:
    source = _TinyConditionedModel(text_width=2)
    target = _TinyConditionedModel(text_width=3)
    checkpoint_path = tmp_path / "legacy-with-ignore.ckpt"
    _save_checkpoint(checkpoint_path, source.state_dict())

    with pytest.raises(RuntimeError, match="--legacy-attention-contract"):
        _load_weights_only_checkpoint(
            target,
            checkpoint_path,
            ignore_prefixes=["text_encoder."],
        )


def test_ignored_text_encoder_allows_intentional_conditioner_contract_change(tmp_path: Path) -> None:
    source = _TinyConditionedModel(text_width=2)
    target = _TinyConditionedModel(text_width=3)
    contract = build_model_architecture_contract(source)
    contract["text_conditioning"]["type"] = "omg.generation.conditions.t5.T5TextEncoder"
    checkpoint_path = tmp_path / "t5-to-muril.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            MODEL_ARCHITECTURE_KEY: contract,
        },
        checkpoint_path,
    )

    _load_weights_only_checkpoint(
        target,
        checkpoint_path,
        strict=True,
        ignore_prefixes=["text_encoder."],
    )


def test_ignored_text_encoder_does_not_bypass_backbone_architecture_contract(tmp_path: Path) -> None:
    source = _TinyConditionedModel(text_width=2)
    target = _TinyConditionedModel(text_width=3)
    contract = build_model_architecture_contract(source)
    contract["attention"]["cross_attention_qk_norm"] = False
    checkpoint_path = tmp_path / "wrong-backbone-contract.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            MODEL_ARCHITECTURE_KEY: contract,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="attention semantics"):
        _load_weights_only_checkpoint(
            target,
            checkpoint_path,
            strict=True,
            ignore_prefixes=["text_encoder."],
        )
