from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn

from omg.core.paths import resolve_repo_path


def normalize_hindi_text(value: str) -> str:
    """Apply the runtime normalization locked by the Hindi adapter plan."""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


class MuRILAdapterTextEncoder(nn.Module):
    """Frozen MuRIL followed by the trainable Hindi-to-OMG residual adapter.

    The returned tensors intentionally match ``FrozenT5TextEncoder`` so the
    validated OMG denoiser remains unchanged: ``context`` is ``[B, 50, 768]``
    and ``mask`` is ``[B, 50]`` for the default configuration.
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        model_revision: str | None = None,
        max_length: int = 50,
        output_dim: int = 768,
        adapter_hidden_dim: int = 3072,
        dropout: float = 0.1,
        residual_scale_init: float = 0.1,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MuRILAdapterTextEncoder requires the `transformers` package. "
                "Install OMG with the training extras."
            ) from exc

        self.model_name = self._resolve_model_name(str(model_name))
        self.model_revision = None if model_revision in {None, "", "null"} else str(model_revision)
        self.max_length = int(max_length)
        self.output_dim = int(output_dim)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if float(residual_scale_init) <= 0.0:
            raise ValueError("residual_scale_init must be positive")

        load_kwargs = {"local_files_only": bool(local_files_only)}
        if self.model_revision is not None:
            load_kwargs["revision"] = self.model_revision
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
            self.encoder = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        except OSError as exc:
            raise OSError(
                f"Failed to load MuRIL from {self.model_name!r}. Download a pinned "
                "google/muril-base-cased snapshot to the configured models directory "
                "or provide a valid Hugging Face model id."
            ) from exc

        hidden_dim = int(getattr(self.encoder.config, "hidden_size", 0))
        if hidden_dim != self.output_dim:
            raise ValueError(
                "The core Hindi adapter requires MuRIL and OMG to share width 768; "
                f"encoder hidden_size={hidden_dim}, output_dim={self.output_dim}"
            )

        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.input_layer_norm = nn.LayerNorm(self.output_dim)
        self.fc1 = nn.Linear(self.output_dim, self.adapter_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(self.adapter_hidden_dim, self.output_dim)
        self.output_layer_norm = nn.LayerNorm(self.output_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        path = Path(model_name).expanduser()
        first_part = path.parts[0] if path.parts else ""
        is_local_path = (
            path.is_absolute()
            or str(path).startswith((".", "~"))
            or first_part in {"models", "assets", "outputs", "checkpoints"}
            or path.exists()
        )
        if not is_local_path:
            return model_name
        resolved = path if path.is_absolute() else resolve_repo_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"MuRIL model path does not exist: {resolved}")
        return str(resolved)

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        for module in (
            self.input_layer_norm,
            self.fc1,
            self.fc2,
            self.output_layer_norm,
        ):
            yield from module.parameters()
        yield self.residual_scale

    def enable_adapter_training(self) -> None:
        self.encoder.requires_grad_(False)
        for parameter in self.adapter_parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> MuRILAdapterTextEncoder:
        super().train(mode)
        # A frozen encoder must not re-enable Dropout when Lightning calls
        # model.train(). Only the small adapter follows ``mode``.
        self.encoder.eval()
        return self

    def _null_output(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        reference = self.residual_scale
        context = torch.zeros(
            batch_size,
            self.max_length,
            self.output_dim,
            device=device,
            dtype=reference.dtype,
        )
        mask = torch.zeros(batch_size, self.max_length, device=device, dtype=torch.bool)
        return {"context": context, "mask": mask}

    def forward(
        self,
        captions: list[str],
        has_text: torch.Tensor | None = None,
        force_null_text: bool = False,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        captions = [normalize_hindi_text(caption) for caption in captions]
        batch_size = len(captions)
        if batch_size == 0:
            raise ValueError("MuRILAdapterTextEncoder received an empty caption batch")
        if device is None:
            device = self.residual_scale.device
        if has_text is None:
            has_text = torch.tensor([bool(caption) for caption in captions], device=device, dtype=torch.bool)
        else:
            if tuple(has_text.shape) != (batch_size,):
                raise ValueError(
                    f"has_text must have shape ({batch_size},), got {tuple(has_text.shape)}"
                )
            has_text = has_text.to(device=device, dtype=torch.bool)
        if force_null_text:
            return self._null_output(batch_size, device)

        tokenized = self.tokenizer(
            captions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            hidden = self.encoder(**tokenized).last_hidden_state

        mask = tokenized["attention_mask"].to(dtype=torch.bool) & has_text.view(-1, 1)
        normalized = self.input_layer_norm(hidden)
        residual = self.fc2(self.dropout(self.activation(self.fc1(normalized))))
        context = self.output_layer_norm(normalized + self.residual_scale.to(normalized.dtype) * residual)
        context = context * mask.unsqueeze(-1).to(context.dtype)
        return {"context": context, "mask": mask}
