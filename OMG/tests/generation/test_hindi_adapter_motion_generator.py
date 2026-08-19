from __future__ import annotations

import torch
import torch.nn as nn

from omg.generation.models.hindi_adapter_motion_generator import HindiAdapterMotionGenerator


class _Representation(nn.Module):
    feat_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(4))

    def normalize_features(self, value):
        return value


class _Denoiser(nn.Module):
    hidden_dim = 8
    frame_cond_injection = "sum_to_time"

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.text_proj = nn.Linear(8, 4, bias=False)
        self.x_seen: list[torch.Tensor] = []
        self.t_seen: list[torch.Tensor] = []

    def forward(self, x, t, conditions, valid_mask=None):
        del valid_mask
        self.x_seen.append(x.detach().clone())
        self.t_seen.append(t.detach().clone())
        mask = conditions["text_mask"].unsqueeze(-1).to(conditions["text_context"].dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        text = (conditions["text_context"] * mask).sum(dim=1) / denominator
        return x + self.text_proj(text)[:, None, :]


class _Diffusion(nn.Module):
    pass


class _Loss(nn.Module):
    weights = {}


class _StudentTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(1, 8, bias=False)
        self.adapter = nn.Linear(8, 8, bias=False)
        self.frozen.requires_grad_(False)

    def adapter_parameters(self):
        yield from self.adapter.parameters()

    def enable_adapter_training(self):
        self.frozen.requires_grad_(False)
        self.adapter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, captions, has_text=None, force_null_text=False, device=None):
        device = device or self.adapter.weight.device
        batch_size = len(captions)
        if force_null_text:
            return {
                "context": torch.zeros(batch_size, 2, 8, device=device),
                "mask": torch.zeros(batch_size, 2, dtype=torch.bool, device=device),
            }
        lengths = torch.tensor([len(value) for value in captions], device=device, dtype=torch.float32)
        hidden = self.frozen(lengths[:, None]).unsqueeze(1).expand(-1, 2, -1)
        context = self.adapter(hidden)
        mask = torch.ones(batch_size, 2, dtype=torch.bool, device=device)
        if has_text is not None:
            mask = mask & has_text.to(device=device, dtype=torch.bool)[:, None]
        return {"context": context * mask.unsqueeze(-1), "mask": mask}


class _TeacherTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, captions, has_text=None, force_null_text=False, device=None):
        del captions, force_null_text
        device = device or self.scale.device
        batch_size = int(has_text.shape[0])
        mask = has_text.to(device=device, dtype=torch.bool)[:, None].expand(-1, 2)
        context = self.scale.to(device) * torch.ones(batch_size, 2, 8, device=device)
        return {"context": context * mask.unsqueeze(-1), "mask": mask}


def _model() -> HindiAdapterMotionGenerator:
    return HindiAdapterMotionGenerator(
        representation=_Representation(),
        denoiser=_Denoiser(),
        diffusion=_Diffusion(),
        loss=_Loss(),
        text_encoder=_StudentTextEncoder(),
        teacher_text_encoder=_TeacherTextEncoder(),
        response_loss_probability=1.0,
        response_loss_initial_weight=0.5,
        response_loss_final_weight=0.1,
        response_loss_hold_steps=5000,
        response_loss_decay_steps=20000,
        condition_dim=8,
        text_mask_prob=0.0,
        history_mask_prob=0.0,
        frame_cond_injection="sum_to_time",
        scheduler=None,
    )


def test_adapter_is_the_only_trainable_component_and_optimizer_group() -> None:
    model = _model()
    expected = {id(parameter) for parameter in model.text_encoder.adapter_parameters()}
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert actual == expected
    optimizer = model.configure_optimizers()
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == expected

    model.train()
    assert not model.denoiser.training
    assert model.text_encoder.adapter.training
    assert not model.text_encoder.frozen.training


def test_response_teacher_reuses_student_noise_timestep_and_is_detached() -> None:
    model = _model()
    model.log = lambda *args, **kwargs: None
    model._response_loss_weight = lambda: 0.5
    batch = {
        "caption": ["आगे चलो", "बायाँ हाथ उठाओ"],
        "teacher_caption": ["walk forward", "raise the left hand"],
        "has_text": torch.ones(2, dtype=torch.bool),
        "history_features": torch.zeros(2, 2, 4),
    }
    conditions = model._conditions(batch)
    x_t = torch.randn(2, 3, 4)
    timesteps = torch.tensor([7, 13])
    t_seq = timesteps[:, None].expand(-1, 3)
    student_prediction = model.denoiser(x_t, t_seq, conditions, valid_mask=torch.ones(2, 3, dtype=torch.bool))
    terms = model._auxiliary_losses(
        batch=batch,
        split="train",
        target=torch.zeros_like(x_t),
        valid=torch.tensor([[True, True, True], [True, True, False]]),
        history_len=0,
        conditions=conditions,
        diffusion_info={
            "x_t": x_t,
            "timesteps": timesteps,
            "raw_prediction": student_prediction,
        },
    )
    assert set(terms) == {"response_loss"}
    assert terms["response_loss"].requires_grad
    assert len(model.denoiser.x_seen) == 2
    torch.testing.assert_close(model.denoiser.x_seen[0], model.denoiser.x_seen[1])
    torch.testing.assert_close(model.denoiser.t_seen[0], model.denoiser.t_seen[1])

    terms["response_loss"].backward()
    assert any(parameter.grad is not None for parameter in model.text_encoder.adapter.parameters())
    assert all(parameter.grad is None for parameter in model.denoiser.parameters())
    teacher = model._teacher_text_encoder
    assert teacher is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not teacher.training


def test_response_loss_is_skipped_when_student_condition_is_dropped() -> None:
    model = _model()
    batch = {
        "teacher_caption": ["walk"],
    }
    terms = model._auxiliary_losses(
        batch=batch,
        split="train",
        target=torch.zeros(1, 2, 4),
        valid=torch.ones(1, 2, dtype=torch.bool),
        history_len=0,
        conditions={
            "text_context": torch.zeros(1, 2, 8),
            "text_mask": torch.zeros(1, 2, dtype=torch.bool),
        },
        diffusion_info={
            "x_t": torch.zeros(1, 2, 4),
            "timesteps": torch.zeros(1, dtype=torch.long),
            "raw_prediction": torch.zeros(1, 2, 4),
        },
    )
    assert set(terms) == {"response_loss"}
    assert float(terms["response_loss"]) == 0.0
    assert model._teacher_text_encoder is None
