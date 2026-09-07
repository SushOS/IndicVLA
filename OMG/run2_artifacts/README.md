# Run 2 artifacts

This directory holds the **small, reviewable** outputs of Run 2 — evaluation JSONs, training
and pipeline logs, and the provenance lock. The large binaries are deliberately not in git.

## What is here

```
logs/                  every stage log, including the failed and ablation runs
runs/                  per-run selection and metric JSON
eval_report.json       the original (superseded) evaluation
eval_fixed_hi.json     corrected evaluation, Hindi arms
eval_fixed_en.json     corrected evaluation, English arms
reference_points.json  English ceiling and text-blind floor, same protocol
baseline_raw_omg.json  stock OMG given Hindi -- the real baseline
adapter_calibration*.json / .pt   measured t5 + MuRIL geometry
RUN_LOCK.json          git state, weight hashes, package versions, GPU
TEARDOWN_MANIFEST.json SHA-256 of every artifact verified on the Hub
```

## What is NOT here, and where to get it

| artifact | size | location |
|---|---|---|
| adapter checkpoints (121 `.pt`) | 2.3 GB | `CodeSushh/maila-run2-motionpivot` on HuggingFace |
| rendered comparison videos (36 `.mp4`) | ~150 MB | same repo, `panels/` and `full_panels/` |
| converted 125-D dataset | 620 MiB | same repo, `data/amass_g1_omg125_converted.tgz` |

Every one of those was re-downloaded from the Hub and SHA-256 verified before the training
box was destroyed — see `TEARDOWN_MANIFEST.json`.

## Reading the evaluation files

**Use `eval_fixed_*.json`, not `eval_report.json`.** The original computed motion-quality
metrics on double-denormalised output: `generate()` returns denormalised features and
`rep.decode()` denormalises again, which compressed motion to ~24% of true scale. That is why
joint-limit, ground-penetration and fall rates all came out as exactly 0.000 — a degenerate
metric, not a good result. SSOT §21 has the measurement.

Caption-sensitivity numbers (`rel`, `rel_draw`, `tiv`, `L_correct`) are **unaffected** in both
files. They come from `native_loss` on the model directly and never touch the decode path.

The most trustworthy motion-quality numbers are in `full_panels/manifest.json` on the Hub:
6 full-length closed-loop clips scored against ground truth read straight from the source
`.npz`, with no codec round-trip at all.

## Headline result

| | caption sensitivity |
|---|---|
| no text (floor) | 0.00% |
| stock OMG + Hindi | +1.80% |
| stock OMG + English (zero-shot) | +9.80% |
| **MAILA + Hindi** | **+10.46%** |
| MAILA + English (control) | +14.69% |

See `../RUN2_SUMMARY.md` for the plain-language version and `../MAILA_SSOT_2026-09-05.md` for
full provenance, including the four measurement bugs found and corrected along the way.
