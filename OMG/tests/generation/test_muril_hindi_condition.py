from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from omg.generation.conditions.muril import MuRILAdapterTextEncoder, normalize_hindi_text


class _FakeTokenizer:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, captions, *, max_length, padding, truncation, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        self.seen = list(captions)
        input_ids = torch.zeros(len(captions), max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, caption in enumerate(captions):
            length = min(max(2, len(caption.split()) + 2), max_length)
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(64, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def _conditioner(monkeypatch, *, hidden_size: int = 8, max_length: int = 5):
    tokenizer = _FakeTokenizer()
    encoder = _FakeEncoder(hidden_size)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *args, **kwargs: encoder,
    )
    conditioner = MuRILAdapterTextEncoder(
        model_name="fake-muril",
        max_length=max_length,
        output_dim=hidden_size,
        adapter_hidden_dim=hidden_size * 4,
        dropout=0.0,
    )
    return conditioner, tokenizer, encoder


def test_hindi_normalization_is_nfc_and_whitespace_only() -> None:
    assert normalize_hindi_text("  आगे\tचलो  \n") == "आगे चलो"


def test_muril_adapter_matches_omg_shape_mask_and_freezing(monkeypatch) -> None:
    conditioner, tokenizer, encoder = _conditioner(monkeypatch)
    output = conditioner(
        ["धीरे आगे चलो", "दायाँ हाथ उठाओ"],
        has_text=torch.tensor([True, False]),
    )
    assert tokenizer.seen == ["धीरे आगे चलो", "दायाँ हाथ उठाओ"]
    assert output["context"].shape == (2, 5, 8)
    assert output["mask"].shape == (2, 5)
    assert output["mask"].dtype == torch.bool
    assert output["context"].requires_grad
    assert not output["mask"][1].any()
    assert torch.count_nonzero(output["context"][1]) == 0
    assert torch.count_nonzero(output["context"][~output["mask"]]) == 0

    output["context"].sum().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in conditioner.adapter_parameters())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    conditioner.train()
    assert conditioner.training
    assert not encoder.training
    assert conditioner.dropout.training


def test_muril_force_null_is_exact_zero_and_skips_tokenization(monkeypatch) -> None:
    conditioner, tokenizer, _ = _conditioner(monkeypatch)
    output = conditioner(
        ["आगे चलो", "पीछे मुड़ो"],
        has_text=torch.ones(2, dtype=torch.bool),
        force_null_text=True,
    )
    assert tokenizer.seen == []
    assert not output["mask"].any()
    assert torch.count_nonzero(output["context"]) == 0


def test_muril_rejects_malformed_has_text(monkeypatch) -> None:
    conditioner, _, _ = _conditioner(monkeypatch)
    try:
        conditioner(["आगे चलो", "रुको"], has_text=torch.ones(2, 1, dtype=torch.bool))
    except ValueError as exc:
        assert "has_text must have shape" in str(exc)
    else:
        raise AssertionError("Expected malformed has_text to fail")
