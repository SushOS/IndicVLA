# VLA Dataset Generation

This project builds a multilingual motion-instruction dataset for a robot/VLA pipeline.

The current workflow is:

1. Generate English motion instructions across 7 tiers.
2. Prepare the English corpus as Kimodo-validated/frozen for downstream use.
3. Translate the English instructions into Hindi, Bengali, and Telugu.
4. Track translation outputs, manifests, summaries, and final release artifacts.

## Project Structure

- `genenrate_prompts.py`
  Generates the English instruction corpus and metadata.
- `prepare_multilingual_translation.py`
  Converts the English corpus into a translation-ready corpus layout.
- `run_llm_translation.py`
  Runs local IndicTrans2 translation for Hindi, Bengali, and Telugu.
- `translation_pipeline_utils.py`
  Shared helpers for JSONL IO, manifests, summaries, and pipeline record updates.
- `finalize_multilingual_corpus.py`
  Finalizes the multilingual corpus after translation and review.
- `corpus_build/`
  Generated dataset artifacts, manifests, summaries, and run-state files.

## Dataset Pipeline

### 1. English generation

`genenrate_prompts.py` generates the base English corpus. Each item includes:

- canonical English instruction
- robot-natural English instruction
- tier
- motion family
- embodiment metadata
- safety / ambiguity metadata

Output:

- `corpus_build/corpus_raw.jsonl`

### 2. Translation-ready preparation

`prepare_multilingual_translation.py` prepares the corpus for multilingual expansion.

It adds:

- translation placeholders for `hi`, `bn`, and `te`
- pipeline stage metadata
- translation manifest rows
- summary statistics

Outputs:

- `corpus_build/corpus_translation_ready.jsonl`
- `corpus_build/translation_manifest.jsonl`
- `corpus_build/translation_summary.json`

### 3. Local translation

`run_llm_translation.py` translates the English instructions locally using IndicTrans2.

Current default translation model:

- `ai4bharat/indictrans2-en-indic-1B`

Built-in optional verifier model:

- `ai4bharat/indictrans2-indic-en-1B`

The verifier-model code path is already implemented in `run_llm_translation.py`. When enabled, it uses Indic-to-English back-translation to estimate:

- action preservation
- order preservation
- direction preservation
- safety preservation

Verification can still be skipped if you want a translation-only run.

Output files updated in place:

- `corpus_build/corpus_translation_ready.jsonl`
- `corpus_build/translation_manifest.jsonl`
- `corpus_build/translation_summary.json`
- `corpus_build/translation_indictrans2_run_state.json`

### 4. Finalization

`finalize_multilingual_corpus.py` promotes translated fields into a release-ready corpus after review.

## Setup

Recommended: use a separate Python environment for translation so model dependencies do not conflict with other local tooling.

Example:

```bash
conda create -n indictrans python=3.10 -y
conda activate indictrans
python -m pip install -r requirements.txt
python -m pip install protobuf sentencepiece IndicTransToolkit
```

Depending on the IndicTrans2 model you use, you may also need Hugging Face access approval and login.

## Usage

### Generate the English corpus

```bash
python genenrate_prompts.py
```

### Prepare the translation-ready corpus

```bash
python prepare_multilingual_translation.py
```

### Run a small translation test

```bash
python run_llm_translation.py --max-batches 1 --batch-size 8 --skip-verification
```

### Run the complete dataset translation without verification

This translates the full English dataset into Hindi, Bengali, and Telugu using the translation model only.

```bash
python run_llm_translation.py --batch-size 32 --skip-verification
```

### Run a small translation test with verification enabled

This uses both models:

- translation: `ai4bharat/indictrans2-en-indic-1B`
- verifier: `ai4bharat/indictrans2-indic-en-1B`

```bash
python run_llm_translation.py --max-batches 1 --batch-size 8
```

### Run the complete dataset translation with verification enabled

Use this if you have access to both the translation model and the verifier model.

```bash
python run_llm_translation.py --batch-size 32
```

### Finalize the multilingual corpus

```bash
python finalize_multilingual_corpus.py
```

## Key Output Files

- `corpus_build/corpus_raw.jsonl`
  Base English corpus.
- `corpus_build/corpus_translation_ready.jsonl`
  Main working corpus with multilingual fields.
- `corpus_build/translation_manifest.jsonl`
  Flattened per-language translation manifest.
- `corpus_build/translation_summary.json`
  Summary counts and pipeline status.
- `corpus_build/translation_indictrans2_run_state.json`
  Translation run progress and model metadata.

## Notes

- The script name `genenrate_prompts.py` is kept as-is to match the current repo.
- Translation verification is heuristic when enabled; human/bilingual review is still the final quality gate.
- Many files in `corpus_build/` are generated artifacts and are ignored by `.gitignore`.
