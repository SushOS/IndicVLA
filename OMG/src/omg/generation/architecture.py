from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_ARCHITECTURE_KEY = "omg_model_architecture"
MODEL_ARCHITECTURE_FORMAT = "omg.model_architecture"
MODEL_ARCHITECTURE_VERSION = 2
SUPPORTED_MODEL_ARCHITECTURE_VERSIONS = {1, MODEL_ARCHITECTURE_VERSION}

LEGACY_ATTENTION_CONTRACTS = {
    "none": {
        "rotary_self_attention_qk_norm": False,
        "cross_attention_qk_norm": False,
    },
    "cross-only": {
        "rotary_self_attention_qk_norm": False,
        "cross_attention_qk_norm": True,
    },
    "self-only": {
        "rotary_self_attention_qk_norm": True,
        "cross_attention_qk_norm": False,
    },
    "self-and-cross": {
        "rotary_self_attention_qk_norm": True,
        "cross_attention_qk_norm": True,
    },
}


def build_model_architecture_contract(model: Any) -> dict[str, Any]:
    denoiser = model.denoiser
    text_encoder = getattr(model, "text_encoder", None)
    text_conditioning = None
    if text_encoder is not None:
        text_conditioning = {
            "type": f"{type(text_encoder).__module__}.{type(text_encoder).__qualname__}",
            "model_name": str(getattr(text_encoder, "model_name", "")),
            "model_revision": getattr(text_encoder, "model_revision", None),
            "max_length": int(getattr(text_encoder, "max_length", 0)),
            "output_dim": int(getattr(text_encoder, "output_dim", 0)),
        }
        if hasattr(text_encoder, "adapter_hidden_dim"):
            text_conditioning["adapter"] = {
                "type": "tokenwise_residual_mlp",
                "hidden_dim": int(text_encoder.adapter_hidden_dim),
            }
    return {
        "format": MODEL_ARCHITECTURE_FORMAT,
        "version": MODEL_ARCHITECTURE_VERSION,
        "denoiser_type": f"{type(denoiser).__module__}.{type(denoiser).__qualname__}",
        "frame_cond_injection": str(getattr(model, "frame_cond_injection", "")),
        "history_pos_encoding": str(getattr(model, "history_pos_encoding", "none")),
        "attention": {
            "rotary_self_attention_qk_norm": bool(
                getattr(denoiser, "self_attention_qk_norm", False)
            ),
            "cross_attention_qk_norm": bool(
                getattr(denoiser, "cross_attention_qk_norm", False)
            ),
        },
        "text_conditioning": text_conditioning,
    }


def _validate_contract_shape(contract: Mapping[str, Any]) -> None:
    if contract.get("format") != MODEL_ARCHITECTURE_FORMAT:
        raise RuntimeError(
            "Unsupported checkpoint architecture format: "
            f"{contract.get('format')!r}; expected {MODEL_ARCHITECTURE_FORMAT!r}"
        )
    version = int(contract.get("version", -1))
    if version not in SUPPORTED_MODEL_ARCHITECTURE_VERSIONS:
        raise RuntimeError(
            "Unsupported checkpoint architecture version: "
            f"{contract.get('version')!r}; expected one of "
            f"{sorted(SUPPORTED_MODEL_ARCHITECTURE_VERSIONS)}"
        )
    if version >= 2 and "history_pos_encoding" not in contract:
        raise RuntimeError("Checkpoint architecture contract is missing history_pos_encoding")
    attention = contract.get("attention")
    if not isinstance(attention, Mapping):
        raise RuntimeError("Checkpoint architecture contract is missing the attention mapping")
    missing = {
        "rotary_self_attention_qk_norm",
        "cross_attention_qk_norm",
    } - set(attention)
    if missing:
        raise RuntimeError(f"Checkpoint architecture attention contract is missing keys: {sorted(missing)}")


def validate_checkpoint_architecture_contract(
    checkpoint: Mapping[str, Any],
    model: Any,
    *,
    legacy_attention_contract: str | None = None,
    validate_text_conditioning: bool = True,
) -> dict[str, Any]:
    actual = build_model_architecture_contract(model)
    recorded = checkpoint.get(MODEL_ARCHITECTURE_KEY)

    if recorded is None:
        if legacy_attention_contract is None:
            choices = ", ".join(LEGACY_ATTENTION_CONTRACTS)
            raise RuntimeError(
                "Checkpoint has no OMG architecture contract. Attention normalization changes no parameter "
                "shapes, so strict state-dict loading cannot detect this semantic mismatch. Re-run with "
                f"--legacy-attention-contract {{{choices}}} and matching Hydra denoiser overrides."
            )
        if legacy_attention_contract not in LEGACY_ATTENTION_CONTRACTS:
            raise ValueError(f"Unknown legacy attention contract: {legacy_attention_contract!r}")
        expected_attention = LEGACY_ATTENTION_CONTRACTS[legacy_attention_contract]
        source = f"legacy declaration {legacy_attention_contract!r}"
    else:
        if not isinstance(recorded, Mapping):
            raise RuntimeError(f"Checkpoint {MODEL_ARCHITECTURE_KEY} must be a mapping")
        _validate_contract_shape(recorded)
        expected_attention = {
            key: bool(recorded["attention"][key])
            for key in (
                "rotary_self_attention_qk_norm",
                "cross_attention_qk_norm",
            )
        }
        source = "checkpoint architecture contract"
        for key in ("denoiser_type", "frame_cond_injection"):
            if str(recorded.get(key, "")) != str(actual[key]):
                raise RuntimeError(
                    f"Instantiated model {key} does not match the checkpoint architecture contract: "
                    f"expected={recorded.get(key)!r}, actual={actual[key]!r}"
                )
        recorded_history_pos_encoding = recorded.get("history_pos_encoding")
        if recorded_history_pos_encoding is not None:
            actual_history_pos_encoding = actual["history_pos_encoding"]
            if str(recorded_history_pos_encoding) != str(actual_history_pos_encoding):
                raise RuntimeError(
                    "Instantiated model history positional encoding does not match the checkpoint "
                    f"architecture contract: expected={recorded_history_pos_encoding!r}, "
                    f"actual={actual_history_pos_encoding!r}."
                )
        recorded_text = recorded.get("text_conditioning")
        if validate_text_conditioning and recorded_text is not None:
            actual_text = actual.get("text_conditioning")
            if not isinstance(recorded_text, Mapping) or not isinstance(actual_text, Mapping):
                raise RuntimeError(
                    "Instantiated model text conditioning does not match the checkpoint architecture contract"
                )
            for key in ("type", "max_length", "output_dim"):
                if recorded_text.get(key) != actual_text.get(key):
                    raise RuntimeError(
                        "Instantiated model text conditioning does not match the checkpoint architecture "
                        f"contract for {key}: expected={recorded_text.get(key)!r}, "
                        f"actual={actual_text.get(key)!r}"
                    )
            recorded_revision = recorded_text.get("model_revision")
            actual_revision = actual_text.get("model_revision")
            if recorded_revision and actual_revision and str(recorded_revision) != str(actual_revision):
                raise RuntimeError(
                    "Instantiated model text conditioning revision does not match the checkpoint "
                    f"architecture contract: expected={recorded_revision!r}, actual={actual_revision!r}"
                )
            if recorded_text.get("adapter") != actual_text.get("adapter"):
                raise RuntimeError(
                    "Instantiated model text adapter does not match the checkpoint architecture contract: "
                    f"expected={recorded_text.get('adapter')!r}, actual={actual_text.get('adapter')!r}"
                )

    if actual["attention"] != expected_attention:
        raise RuntimeError(
            "Instantiated denoiser attention semantics do not match the "
            f"{source}: expected={expected_attention}, actual={actual['attention']}. "
            "Set denoiser.self_attention_qk_norm and denoiser.cross_attention_qk_norm to the checkpoint's "
            "training values before exporting."
        )
    return actual
