#!/usr/bin/env python3
"""MAILA Hindi conditioner: frozen MuRIL -> trainable residual adapter -> OMG's text contract.

Drop-in replacement for `omg.generation.conditions.t5.FrozenT5TextEncoder`, swapped in
without forking OMG:

    model.text_encoder._target_=maila_encoder.MurilAdapterEncoder
    model.text_encoder.adapter_ckpt=/path/adapter.pt
    model.text_encoder.muril_name=/workspace/models/muril-base-cased

Contract it must satisfy (read from t5.py:88-103):
    forward(captions, has_text=None, force_null_text=False, device=None)
        -> {"context": (B, 50, 768) float, "mask": (B, 50) bool}

THREE INVARIANTS THIS CLASS EXISTS TO PROTECT
---------------------------------------------
1. OMG's `proj` is `nn.Identity()` -- there is NO learned layer between the text encoder
   and cross-attention, so the adapter output must land in t5-base's ACTUAL geometry.
2. `force_null_text=True` returns t5-base's OWN null context, computed once at construction
   and cached as a buffer. OMG generates at cfg_scale=2.5 and the unconditional branch must
   stay byte-identical to what the frozen DiT was trained with. Routing "" through the
   adapter would silently redefine guidance.
3. `has_text` is applied exactly as OMG does: mask &= has_text, then zero the context.

`adapter_ckpt=None` degrades to plain t5-base passthrough, so constructing with defaults
reproduces today's behaviour exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import torch
import torch.nn as nn


@dataclass
class AdapterConfig:
    width: int = 768          # MuRIL hidden == t5-base d_model == OMG text_dim
    mlp_ratio: float = 4.0    # -> 3072 hidden units (MAILA plan 5.2)
    dropout: float = 0.1
    alpha_init: float = 0.1   # learnable residual scale
    calibration: str | None = None   # t5 per-channel {mu,sd}; see _init_out_norm
    norm_mode: str = "layernorm"     # "layernorm" (as-built) | "corpus" (see forward)


class ResidualAdapter(nn.Module):
    """U = LN(H); R = W2(drop(gelu(W1(U)))); Z = outLN(U + alpha * R)   (MAILA plan 5.2)

    Applied independently per token: MuRIL already contextualises, so a second transformer
    is unnecessary for the first experiment.

    NOTE on `alpha`: initialised small (0.1) so training starts near normalised-MuRIL and
    moves outward. MuRIL space is NOT t5 space, so unlike a same-space residual there is no
    "safe" identity init -- alpha must grow for this to work. Log it; if it saturates or the
    tiny-set overfit in Phase 1 fails, the residual framing is underpowered and the identity
    path should be replaced by a full linear map.
    """

    def __init__(self, cfg: AdapterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AdapterConfig()
        w = self.cfg.width
        hidden = int(w * self.cfg.mlp_ratio)
        self.in_norm = nn.LayerNorm(w)
        self.fc1 = nn.Linear(w, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.cfg.dropout)
        self.fc2 = nn.Linear(hidden, w)
        self.out_norm = nn.LayerNorm(w)
        self.alpha = nn.Parameter(torch.tensor(float(self.cfg.alpha_init)))
        # source-side corpus statistics, used only by norm_mode="corpus". Buffers, not
        # parameters: they are frozen measurements of MuRIL, not something to learn.
        self.register_buffer("src_mu", torch.zeros(w))
        self.register_buffer("src_sd", torch.ones(w))
        self._init_out_norm()

    def _init_out_norm(self) -> None:
        """Start out_norm in t5-base's ACTUAL per-channel geometry, not at w=1 / b=0.

        MEASURED 2026-09-06 (calibrate_t5_geometry.py + 1,024 val windows, replayed seeds):

            adapter state                        L_correct   L_wrong       rel   tok_norm
            random init, out_norm w=1 b=0          0.07454   0.07338    -1.56%     27.713
            random init, out_norm RECOLOURED       0.03883   0.03884    +0.04%      7.396
            t5-base(english), the DiT's reference        -         -         -      6.758

        nn.LayerNorm's default affine pins every output token to unit variance, so
        ||token|| == sqrt(768) == 27.71 REGARDLESS of alpha or of anything the MLP learns.
        t5-base's real contexts have norm 6.76. OMG's `proj` is nn.Identity(), so no layer
        exists in between to absorb the difference: the frozen DiT reads context vectors
        4.1x too long, and HALF the initial denoising loss is that scale error, not semantics.

        The 2026-09-05 lambda_en=0.1 run spent 8,000 steps paying it off -- L fell
        0.0745 -> 0.0257 while cos(per-channel mean, t5) crawled 0.043 -> 0.133 and `rel`
        never left zero. The optimiser was grinding 768 LayerNorm gains toward 0.25 instead
        of learning Hindi -> motion.

        LayerNorm already emits zero-mean/unit-variance z and then applies weight/bias, so
        weight[d] = std_t5[d], bias[d] = mean_t5[d] simply RECOLOURS z into t5's marginal
        distribution. It costs no parameters and no capacity -- it only moves the starting
        point inside the distribution the frozen DiT was fitted on.

        This must be an INIT, not a repair: recolouring a checkpoint already trained at the
        wrong scale made it WORSE (rel +1.10% -> +0.30%), because those weights had adapted
        to 27.7. Calibration is measured on TRAIN-split English captions only.
        """
        if not self.cfg.calibration:
            if self.cfg.norm_mode == "corpus":
                raise ValueError('norm_mode="corpus" requires --t5-calibration: the source '
                                 "statistics live in that file")
            return
        st = torch.load(self.cfg.calibration, map_location="cpu", weights_only=False)
        mu, sd = st["mu"].float(), st["sd"].float()
        if tuple(mu.shape) != (self.cfg.width,) or tuple(sd.shape) != (self.cfg.width,):
            raise ValueError(f"calibration shape {tuple(mu.shape)} != ({self.cfg.width},)")
        with torch.no_grad():
            self.out_norm.weight.copy_(sd)
            self.out_norm.bias.copy_(mu)
        if self.cfg.norm_mode != "corpus":
            return
        if "src_mu" not in st:
            raise ValueError(f"{self.cfg.calibration} has no MuRIL source statistics; "
                             "re-run calibrate_adapter_geometry.py with --langs")
        with torch.no_grad():
            self.src_mu.copy_(st["src_mu"].float())
            self.src_sd.copy_(st["src_sd"].float().clamp_min(1e-6))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """norm_mode="layernorm" is the as-built path. "corpus" is the measured fix.

        WHY LayerNorm IS THE WRONG NORMALISER HERE (measured 2026-09-06, 512 val captions)

            init                                  btwn/mu   cos(i,j)   L_correct
            LayerNorm + t5 recolour                0.0581     0.9966     0.03005
            centre + per-channel std + recolour    0.8042     0.6080     0.02778
            ZCA whiten + recolour                  0.8925     0.5628     0.02807
            t5-base(english), the DiT's reference  0.7201     0.6563          --

        `cos(i,j)` is the mean cosine between the contexts of two DIFFERENT captions.
        Through the as-built adapter it is 0.9966: every caption produces very nearly the
        SAME vector, so the frozen cross-attention is conditioned on a constant and `rel`
        and `tiv` sit at zero no matter how long training runs.

        The cause is MuRIL's anisotropy. nn.LayerNorm standardises WITHIN a token, across
        the 768 channels, so a direction shared by ALL tokens passes through untouched --
        it is rescaled along with the signal, never removed. Measured ||mu_corpus|| = 26.91
        against a token norm of 27.71: the shared component was essentially the whole token,
        leaving ~0.42 units of caption-specific content against ~7.3 units of constant.

        Subtracting the CORPUS mean (a statistic across tokens, which LayerNorm cannot see)
        removes it, and dividing by the corpus per-channel std equalises the directions that
        remain. Recolouring by t5's own {mu, sd} then places the result in the distribution
        the frozen DiT was fitted on. That recovers 13.8x the caption signal.

        Full ZCA (option C) decorrelates slightly harder but overshoots t5's own anisotropy,
        and after centring the top eigenvalue holds just 9.9% of the variance -- the mean WAS
        the anisotropy, so decorrelation buys almost nothing for its extra risk.

        NOTE this deliberately leaves the output scale free to drift as `alpha` and the MLP
        grow. LayerNorm pinned it, which is what stopped the residual branch from ever
        gaining authority over the identity path (Run 1: alpha moved 0.100 -> 0.131 in
        25,000 steps). Here the branch can actually take over.
        """
        u = self.in_norm(h)
        r = self.fc2(self.drop(self.act(self.fc1(u))))
        y = u + self.alpha * r
        if self.cfg.norm_mode == "corpus":
            z = (y - self.src_mu) / self.src_sd
            return z * self.out_norm.weight + self.out_norm.bias
        return self.out_norm(y)


def apply_has_text(context: torch.Tensor, mask: torch.Tensor,
                   has_text: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim from OMG t5.py:99-102."""
    if has_text is None:
        return context, mask
    has_text = has_text.to(device=context.device, dtype=torch.bool).view(-1, 1)
    mask = mask & has_text
    context = context * mask.unsqueeze(-1).to(context.dtype)
    return context, mask


class MurilAdapterEncoder(nn.Module):
    def __init__(
        self,
        muril_name: str = "google/muril-base-cased",
        t5_model_name: str = "models/t5-base-local",
        adapter_ckpt: str | None = None,
        max_length: int = 50,
        output_dim: int = 768,
        freeze_muril: bool = True,
        adapter: AdapterConfig | None = None,
        t5_calibration: str | None = None,
        norm_mode: str = "layernorm",
        passthrough_t5: bool = False,
    ) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.output_dim = int(output_dim)
        self.passthrough_t5 = bool(passthrough_t5)

        # --- cache t5-base's null context ONCE; CFG must not change (invariant 2)
        from omg.generation.conditions.t5 import FrozenT5TextEncoder
        t5 = FrozenT5TextEncoder(model_name=t5_model_name, max_length=max_length,
                                 output_dim=output_dim).eval()
        with torch.no_grad():
            null = t5([""], device=torch.device("cpu"))
        self.register_buffer("null_context", null["context"].clone())
        self.register_buffer("null_mask", null["mask"].clone())
        self._t5 = t5 if passthrough_t5 else None      # kept only for the control arm

        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(muril_name)
        self.muril = AutoModel.from_pretrained(muril_name)
        if freeze_muril:
            self.muril.eval()
            for p in self.muril.parameters():
                p.requires_grad_(False)

        h = int(self.muril.config.hidden_size)
        if h != output_dim:
            raise ValueError(f"MuRIL hidden {h} != OMG text width {output_dim}; "
                             "a projection would be needed and none is configured")
        acfg = adapter or AdapterConfig()
        if t5_calibration:
            acfg = replace(acfg, calibration=t5_calibration)
        if norm_mode != "layernorm":
            acfg = replace(acfg, norm_mode=norm_mode)
        self.adapter = ResidualAdapter(acfg)
        if adapter_ckpt:
            state = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
            self.adapter.load_state_dict(state["adapter"])

    # ------------------------------------------------------------------ helpers
    def trainable_parameters(self):
        return [p for p in self.adapter.parameters() if p.requires_grad]

    @torch.no_grad()
    def _encode_muril(self, captions: list[str], device: torch.device):
        tok = self.tokenizer(list(captions), max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.muril(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        return out.last_hidden_state, tok["attention_mask"].bool()

    # ------------------------------------------------------------------ contract
    def forward(
        self,
        captions: list[str],
        has_text: torch.Tensor | None = None,
        force_null_text: bool = False,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(captions) == 0:
            raise ValueError("MurilAdapterEncoder received an empty caption batch")
        if device is None:
            device = self.null_context.device
        b = len(captions)

        if force_null_text:
            ctx = self.null_context.to(device).expand(b, -1, -1).clone()
            msk = self.null_mask.to(device).expand(b, -1).clone()
            return dict(zip(("context", "mask"), apply_has_text(ctx, msk, has_text)))

        if self.passthrough_t5:                       # English control arm
            return self._t5(captions, has_text=has_text, device=device)

        hidden, mask = self._encode_muril(captions, device)   # MuRIL frozen -> no grad
        ctx = self.adapter(hidden)                            # <- the only trainable path
        return dict(zip(("context", "mask"), apply_has_text(ctx, mask, has_text)))

    def save_adapter(self, path: str, **extra) -> None:
        torch.save({"adapter": self.adapter.state_dict(),
                    "adapter_config": asdict(self.adapter.cfg), **extra}, path)
