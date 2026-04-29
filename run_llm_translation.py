import argparse
import inspect
import importlib.util
import sys
import types
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from translation_pipeline_utils import (
    LANGUAGES,
    VERIFICATION_FIELDS,
    build_manifest_rows,
    chunked,
    load_jsonl,
    normalize_space,
    summarize_translation_corpus,
    update_entry_with_model_translation,
    write_json,
    write_jsonl,
)

LANGUAGE_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
}

DEFAULT_EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
DEFAULT_INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
TRANSLATION_METHOD = "local_indictrans2"

ACTION_PHRASES = (
    "walk",
    "move",
    "go",
    "step",
    "proceed",
    "advance",
    "turn",
    "rotate",
    "pivot",
    "spin",
    "stop",
    "halt",
    "freeze",
    "pause",
    "hold",
    "crouch",
    "kneel",
    "stand up",
    "rise",
    "lean",
    "bow",
    "sit down",
    "duck",
    "bend",
    "wave",
    "point",
    "raise",
    "lower",
    "clap",
    "nod",
    "salute",
    "follow",
    "pass",
    "yield",
    "inspect",
    "wait",
    "slow down",
    "retreat",
)

DIRECTION_PHRASES = (
    "left",
    "right",
    "forward",
    "backward",
    "ahead",
    "back",
    "around",
    "through",
    "over",
    "under",
    "toward",
    "towards",
    "away",
    "near",
    "beside",
    "in front of",
)

SAFETY_PHRASES = (
    "safe",
    "unsafe",
    "stop",
    "wait",
    "yield",
    "inspect",
    "carefully",
    "cautiously",
    "slowly",
    "slow down",
    "if",
    "when",
    "uncertain",
    "obstacle",
    "blocked",
    "clear",
    "person",
    "pedestrian",
    "distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a frozen Kimodo-cleared English corpus into Hindi, Bengali, "
            "and Telugu locally using IndicTrans2."
        )
    )
    parser.add_argument(
        "--input",
        default="corpus_build/corpus_translation_ready.jsonl",
        help="Translation-ready corpus JSONL.",
    )
    parser.add_argument(
        "--output",
        default="corpus_build/corpus_translation_ready.jsonl",
        help="Updated corpus JSONL. Reuse the same path for resumable updates.",
    )
    parser.add_argument(
        "--manifest-output",
        default="corpus_build/translation_manifest.jsonl",
        help="Manifest JSONL refreshed after each successful batch.",
    )
    parser.add_argument(
        "--summary-output",
        default="corpus_build/translation_summary.json",
        help="Summary JSON refreshed after each successful batch.",
    )
    parser.add_argument(
        "--state-output",
        default="corpus_build/translation_indictrans2_run_state.json",
        help="Run state JSON output path.",
    )
    parser.add_argument(
        "--en-indic-model",
        default=DEFAULT_EN_INDIC_MODEL,
        help="IndicTrans2 model for English -> Indic translation.",
    )
    parser.add_argument(
        "--indic-en-model",
        default=DEFAULT_INDIC_EN_MODEL,
        help="IndicTrans2 model for Indic -> English back-translation verification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of prompts to process per translation batch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of batches to process in this run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype to request when loading the models.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generation length for IndicTrans2 decoding.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam width for IndicTrans2 generation.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip back-translation verification and mark verification checks as null-safe defaults.",
    )
    parser.add_argument(
        "--allow-verification-fallback",
        action="store_true",
        default=True,
        help="If the Indic->English verifier model is unavailable, continue translation without it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without loading models.",
    )
    return parser.parse_args()


def require_dependency(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Missing dependency '{module_name}' for interpreter '{sys.executable}'. "
            f"Install it into this interpreter with: {sys.executable} -m pip install {install_hint.removeprefix('pip install ')}"
        )


def load_indic_processor_class() -> Any:
    package_spec = importlib.util.find_spec("IndicTransToolkit")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ModuleNotFoundError(
            "Missing dependency 'IndicTransToolkit'. Install it into this interpreter with: "
            f"{sys.executable} -m pip install IndicTransToolkit"
        )

    package_dir = Path(next(iter(package_spec.submodule_search_locations)))
    candidate_paths = []
    candidate_paths.extend(sorted(package_dir.glob("processor*.so")))
    candidate_paths.extend(sorted(package_dir.glob("processor*.pyd")))
    candidate_paths.extend(sorted(package_dir.glob("processor.py")))

    last_error = None
    for candidate_path in candidate_paths:
        try:
            module_name = (
                "processor"
                if candidate_path.suffix in {".so", ".pyd"}
                else "_indictrans_processor_direct"
            )
            module_spec = importlib.util.spec_from_file_location(
                module_name,
                candidate_path,
            )
            if module_spec is None or module_spec.loader is None:
                continue
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            processor_cls = getattr(module, "IndicProcessor", None)
            if processor_cls is not None:
                return processor_cls
        except Exception as exc:
            last_error = exc

    raise ImportError(
        "Could not import IndicProcessor from IndicTransToolkit. "
        "This is usually caused by a package/transformers compatibility issue."
    ) from last_error


def resolve_runtime_imports() -> tuple[Any, Any, Any, Any]:
    require_dependency("torch", "pip install torch")
    require_dependency("transformers", "pip install transformers sentencepiece")
    require_dependency(
        "IndicTransToolkit",
        "pip install IndicTransToolkit",
    )

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer
    IndicProcessor = load_indic_processor_class()

    patch_tokenizer_special_token_setattr(PreTrainedTokenizer)

    return torch, AutoModelForSeq2SeqLM, AutoTokenizer, IndicProcessor


def patch_tokenizer_special_token_setattr(PreTrainedTokenizer: Any) -> None:
    if getattr(PreTrainedTokenizer, "_indictrans_lazy_special_tokens_patch", False):
        return

    original_setattr = PreTrainedTokenizer.__setattr__

    def patched_setattr(self: Any, key: str, value: Any) -> None:
        special_token_attrs = (
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens",
        )
        if key in special_token_attrs and "_special_tokens_map" not in self.__dict__:
            object.__setattr__(self, "_special_tokens_map", {})
        original_setattr(self, key, value)

    PreTrainedTokenizer.__setattr__ = patched_setattr
    PreTrainedTokenizer._indictrans_lazy_special_tokens_patch = True


def is_access_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "gated repo",
        "401 client error",
        "403 client error",
        "not in the authorized list",
        "access to model",
        "please log in",
    )
    return any(needle in text for needle in needles)


def is_network_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "nodename nor servname provided",
        "temporary failure in name resolution",
        "name or service not known",
        "failed to establish a new connection",
        "connection error",
        "cannot send a request, as the client has been closed",
        "offline mode",
    )
    return any(needle in text for needle in needles)


def resolve_local_model_source(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name

    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def load_local_indictrans_classes(model_source: str) -> tuple[Any, Any]:
    source_path = Path(model_source)
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Local model source not found: {model_source}")

    ensure_transformers_onnx_shim()

    package_name = "_indictrans_local_bundle"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_path)]
        sys.modules[package_name] = package

    def load_submodule(module_stem: str) -> Any:
        module_name = f"{package_name}.{module_stem}"
        if module_name in sys.modules:
            return sys.modules[module_name]

        module_path = source_path / f"{module_stem}.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    tokenization_module = load_submodule("tokenization_indictrans")
    configuration_module = load_submodule("configuration_indictrans")
    modeling_module = load_submodule("modeling_indictrans")
    patch_indictrans_model_class(getattr(modeling_module, "IndicTransForConditionalGeneration"))
    return (
        getattr(tokenization_module, "IndicTransTokenizer"),
        getattr(modeling_module, "IndicTransForConditionalGeneration"),
    )


def ensure_transformers_onnx_shim() -> None:
    if "transformers.onnx" in sys.modules and "transformers.onnx.utils" in sys.modules:
        return

    try:
        import transformers.onnx  # type: ignore  # noqa: F401
        import transformers.onnx.utils  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    onnx_module = types.ModuleType("transformers.onnx")
    utils_module = types.ModuleType("transformers.onnx.utils")

    class _OnnxConfig:
        pass

    class _OnnxSeq2SeqConfigWithPast(_OnnxConfig):
        pass

    def compute_effective_axis_dimension(*args: Any, **kwargs: Any) -> int:
        if args:
            for value in reversed(args):
                if isinstance(value, int):
                    return value
        return 0

    onnx_module.OnnxConfig = _OnnxConfig
    onnx_module.OnnxSeq2SeqConfigWithPast = _OnnxSeq2SeqConfigWithPast
    utils_module.compute_effective_axis_dimension = compute_effective_axis_dimension

    sys.modules["transformers.onnx"] = onnx_module
    sys.modules["transformers.onnx.utils"] = utils_module


def patch_indictrans_model_class(model_class: Any) -> None:
    if getattr(model_class, "_indictrans_transformers_compat_patch", False):
        return

    tie_weights = getattr(model_class, "tie_weights", None)
    if tie_weights is not None:
        signature = inspect.signature(tie_weights)
        if "recompute_mapping" not in signature.parameters:
            def patched_tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
                return tie_weights(self)
            model_class.tie_weights = patched_tie_weights

    model_class._indictrans_transformers_compat_patch = True


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA was requested but is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


@dataclass
class ModelBundle:
    model_name: str
    model: Any
    tokenizer: Any
    processor: Any
    device: str
    torch: Any


def load_indictrans_model(
    *,
    torch: Any,
    AutoModelForSeq2SeqLM: Any,
    AutoTokenizer: Any,
    IndicProcessor: Any,
    model_name: str,
    device: str,
    dtype: Any,
) -> ModelBundle:
    model_source = resolve_local_model_source(model_name)
    using_local_bundle = Path(model_source).exists()
    if using_local_bundle:
        tokenizer_class, model_class = load_local_indictrans_classes(model_source)
        tokenizer = tokenizer_class.from_pretrained(model_source)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        except Exception as exc:
            if is_network_error(exc):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            else:
                raise

    model_kwargs = {
        "torch_dtype": dtype,
    }
    if not using_local_bundle:
        model_kwargs["trust_remote_code"] = True
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    def build_model(load_kwargs: dict[str, Any]) -> Any:
        if using_local_bundle:
            return model_class.from_pretrained(model_source, **load_kwargs).to(device)
        return AutoModelForSeq2SeqLM.from_pretrained(model_source, **load_kwargs).to(device)

    try:
        model = build_model(model_kwargs)
    except Exception as exc:
        if device == "cuda" and "flash_attention_2" in str(exc):
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            model = build_model(fallback_kwargs)
        elif is_network_error(exc):
            offline_kwargs = dict(model_kwargs)
            offline_kwargs["local_files_only"] = True
            offline_kwargs.pop("attn_implementation", None)
            model = build_model(offline_kwargs)
        else:
            raise

    processor = IndicProcessor(inference=True)
    return ModelBundle(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        torch=torch,
    )


def run_translation(
    bundle: ModelBundle,
    *,
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_beams: int,
) -> list[str]:
    if not sentences:
        return []

    batch = bundle.processor.preprocess_batch(
        sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    inputs = bundle.tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(bundle.device)

    with bundle.torch.no_grad():
        generated_tokens = bundle.model.generate(
            **inputs,
            use_cache=False,
            min_length=0,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,
        )

    decoded = bundle.tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    return [
        normalize_space(text)
        for text in bundle.processor.postprocess_batch(
            [sanitize_decoded_text(text, tgt_lang=tgt_lang) for text in decoded],
            lang=tgt_lang,
        )
    ]


def sanitize_decoded_text(text: str, *, tgt_lang: str) -> str:
    cleaned = text
    for token in ("<s>", "</s>", "<pad>", "<unk>"):
        cleaned = cleaned.replace(token, " ")

    for lang_code in LANGUAGE_CODES.values():
        cleaned = cleaned.replace(lang_code, " ")
    cleaned = cleaned.replace(tgt_lang.replace("_", " "), " ")

    return normalize_space(cleaned)


def normalize_english_for_match(text: str) -> str:
    lowered = normalize_space(text).casefold()
    replacements = {
        "towards": "toward",
        "right away": "",
        "immediately": "",
        "promptly": "",
        "straight away": "",
        "at once": "",
        "directly": "",
        "robot natural": "",
    }
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    return normalize_space(lowered)


def extract_present_phrases(text: str, candidates: tuple[str, ...]) -> list[str]:
    lowered = normalize_english_for_match(text)
    present = []
    for phrase in candidates:
        if phrase in lowered:
            present.append(phrase)
    return present


def check_sequence_preservation(source: str, candidate: str, phrases: tuple[str, ...]) -> bool:
    source_present = extract_present_phrases(source, phrases)
    if not source_present:
        return True
    candidate_norm = normalize_english_for_match(candidate)
    positions = []
    for phrase in source_present:
        pos = candidate_norm.find(phrase)
        if pos < 0:
            return False
        positions.append(pos)
    return positions == sorted(positions)


def check_set_preservation(source: str, candidate: str, phrases: tuple[str, ...]) -> bool:
    source_present = set(extract_present_phrases(source, phrases))
    if not source_present:
        return True
    candidate_present = set(extract_present_phrases(candidate, phrases))
    return source_present.issubset(candidate_present)


def build_verification(
    *,
    canonical_en: str,
    robot_natural_en: str,
    literal_backtranslated: str,
    robot_backtranslated: str,
) -> dict[str, bool]:
    order_ok = check_sequence_preservation(
        canonical_en,
        literal_backtranslated,
        ACTION_PHRASES + DIRECTION_PHRASES,
    ) and check_sequence_preservation(
        robot_natural_en,
        robot_backtranslated,
        ACTION_PHRASES + DIRECTION_PHRASES,
    )

    action_ok = check_set_preservation(
        canonical_en,
        literal_backtranslated,
        ACTION_PHRASES,
    ) and check_set_preservation(
        robot_natural_en,
        robot_backtranslated,
        ACTION_PHRASES,
    )

    direction_ok = check_set_preservation(
        canonical_en,
        literal_backtranslated,
        DIRECTION_PHRASES,
    ) and check_set_preservation(
        robot_natural_en,
        robot_backtranslated,
        DIRECTION_PHRASES,
    )

    safety_ok = check_set_preservation(
        canonical_en,
        literal_backtranslated,
        SAFETY_PHRASES,
    ) and check_set_preservation(
        robot_natural_en,
        robot_backtranslated,
        SAFETY_PHRASES,
    )

    return {
        "action_preservation": action_ok,
        "order_preservation": order_ok,
        "direction_preservation": direction_ok,
        "safety_preservation": safety_ok,
    }


def build_translation_notes(
    *,
    verification: dict[str, bool],
    literal_backtranslated: str | None,
    robot_backtranslated: str | None,
    verification_enabled: bool,
) -> list[str]:
    notes = []
    if verification_enabled:
        notes.append("Translated locally with IndicTrans2.")
        notes.append("Verification was estimated via IndicTrans2 back-translation to English.")
        notes.append(f"Literal back-translation: {literal_backtranslated}")
        notes.append(f"Robot-natural back-translation: {robot_backtranslated}")
        failed = [field for field in VERIFICATION_FIELDS if not verification[field]]
        if failed:
            notes.append(f"Automatic verification flagged: {', '.join(failed)}.")
    else:
        notes.append("Translated locally with IndicTrans2.")
        notes.append("Automatic back-translation verification was skipped for this run.")
    return notes


def build_state_payload(
    *,
    corpus: list[dict[str, Any]],
    translation_model: str,
    verification_model: str | None,
    completed_batches: int,
    total_batches: int,
) -> dict[str, Any]:
    translated = 0
    pending = 0
    for entry in corpus:
        for code in LANGUAGES:
            if language_block_is_complete(entry[f"translation_{code}"]):
                translated += 1
            else:
                pending += 1

    return {
        "translation_method": TRANSLATION_METHOD,
        "translation_model": translation_model,
        "verification_model": verification_model,
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "translated_language_jobs": translated,
        "pending_language_jobs": pending,
    }


def save_progress(
    *,
    corpus: list[dict[str, Any]],
    output_path: str,
    manifest_output: str,
    summary_output: str,
    state_output: str,
    translation_model: str,
    verification_model: str | None,
    completed_batches: int,
    total_batches: int,
) -> None:
    write_jsonl(output_path, corpus)
    write_jsonl(manifest_output, build_manifest_rows(corpus))
    write_json(
        summary_output,
        summarize_translation_corpus(
            corpus,
            assumptions={
                "english_source_frozen": True,
                "kimodo_validation_complete": True,
                "translation_method": TRANSLATION_METHOD,
                "translation_model": translation_model,
                "verification_model": verification_model,
            },
        ),
    )
    write_json(
        state_output,
        build_state_payload(
            corpus=corpus,
            translation_model=translation_model,
            verification_model=verification_model,
            completed_batches=completed_batches,
            total_batches=total_batches,
        ),
    )


def language_block_is_complete(block: dict[str, Any]) -> bool:
    return (
        block.get("status") == "model_translated"
        and bool((block.get("literal") or "").strip())
        and bool((block.get("robot_natural") or "").strip())
    )


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            {
                "input": args.input,
                "output": args.output,
                "en_indic_model": args.en_indic_model,
                "indic_en_model": None if args.skip_verification else args.indic_en_model,
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "device": args.device,
                "dtype": args.dtype,
                "translation_method": TRANSLATION_METHOD,
                "allow_verification_fallback": args.allow_verification_fallback,
            }
        )
        return

    torch, AutoModelForSeq2SeqLM, AutoTokenizer, IndicProcessor = resolve_runtime_imports()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)

    corpus = load_jsonl(args.input)
    pending_entries = [
        entry for entry in corpus
        if any(not language_block_is_complete(entry[f"translation_{code}"]) for code in LANGUAGES)
    ]
    batches = chunked(pending_entries, args.batch_size)
    if args.max_batches is not None:
        batches = batches[:args.max_batches]

    total_batches = len(batches)
    if total_batches == 0:
        print("No pending translations found.")
        return

    en_indic_bundle = load_indictrans_model(
        torch=torch,
        AutoModelForSeq2SeqLM=AutoModelForSeq2SeqLM,
        AutoTokenizer=AutoTokenizer,
        IndicProcessor=IndicProcessor,
        model_name=args.en_indic_model,
        device=device,
        dtype=dtype,
    )
    indic_en_bundle = None
    if not args.skip_verification:
        try:
            indic_en_bundle = load_indictrans_model(
                torch=torch,
                AutoModelForSeq2SeqLM=AutoModelForSeq2SeqLM,
                AutoTokenizer=AutoTokenizer,
                IndicProcessor=IndicProcessor,
                model_name=args.indic_en_model,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            if args.allow_verification_fallback and is_access_error(exc):
                print(
                    "Verification model could not be loaded; "
                    "continuing without back-translation verification."
                )
                print(f"Verification load error          : {exc}")
                indic_en_bundle = None
            else:
                raise

    entry_index = {entry["prompt_id"]: entry for entry in corpus}
    completed_batches = 0

    for batch_number, batch in enumerate(batches, start=1):
        canonical_sentences = [entry["instruction_en"]["canonical"] for entry in batch]
        robot_natural_sentences = [entry["instruction_en"]["robot_natural"] for entry in batch]

        for language_code in LANGUAGES:
            target_lang = LANGUAGE_CODES[language_code]
            literal_outputs = run_translation(
                en_indic_bundle,
                sentences=canonical_sentences,
                src_lang=LANGUAGE_CODES["en"],
                tgt_lang=target_lang,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
            robot_outputs = run_translation(
                en_indic_bundle,
                sentences=robot_natural_sentences,
                src_lang=LANGUAGE_CODES["en"],
                tgt_lang=target_lang,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )

            if indic_en_bundle is not None:
                literal_backtranslations = run_translation(
                    indic_en_bundle,
                    sentences=literal_outputs,
                    src_lang=target_lang,
                    tgt_lang=LANGUAGE_CODES["en"],
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                )
                robot_backtranslations = run_translation(
                    indic_en_bundle,
                    sentences=robot_outputs,
                    src_lang=target_lang,
                    tgt_lang=LANGUAGE_CODES["en"],
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                )
            else:
                literal_backtranslations = [None] * len(batch)
                robot_backtranslations = [None] * len(batch)

            for entry, literal_text, robot_text, literal_back, robot_back in zip(
                batch,
                literal_outputs,
                robot_outputs,
                literal_backtranslations,
                robot_backtranslations,
            ):
                if indic_en_bundle is not None:
                    verification = build_verification(
                        canonical_en=entry["instruction_en"]["canonical"],
                        robot_natural_en=entry["instruction_en"]["robot_natural"],
                        literal_backtranslated=literal_back or "",
                        robot_backtranslated=robot_back or "",
                    )
                else:
                    verification = {field: True for field in VERIFICATION_FIELDS}

                update_entry_with_model_translation(
                    entry_index[entry["prompt_id"]],
                    language_code=language_code,
                    literal=literal_text,
                    robot_natural=robot_text,
                    verification=verification,
                    translation_notes=build_translation_notes(
                        verification=verification,
                        literal_backtranslated=literal_back,
                        robot_backtranslated=robot_back,
                        verification_enabled=indic_en_bundle is not None,
                    ),
                    model=args.en_indic_model,
                    translation_method=TRANSLATION_METHOD,
                )

        completed_batches += 1
        save_progress(
            corpus=corpus,
            output_path=args.output,
            manifest_output=args.manifest_output,
            summary_output=args.summary_output,
            state_output=args.state_output,
            translation_model=args.en_indic_model,
            verification_model=None if args.skip_verification else args.indic_en_model,
            completed_batches=completed_batches,
            total_batches=total_batches,
        )
        print(f"Completed batch {batch_number}/{total_batches} ({len(batch)} prompts).")

    print(f"Updated corpus written to          : {args.output}")
    print(f"Updated translation manifest      : {args.manifest_output}")
    print(f"Updated translation summary       : {args.summary_output}")
    print(f"Updated run state                 : {args.state_output}")
    print(f"Translation model used            : {args.en_indic_model}")
    if args.skip_verification:
        print("Verification model used           : skipped")
    else:
        print(f"Verification model used           : {args.indic_en_model}")


if __name__ == "__main__":
    main()
