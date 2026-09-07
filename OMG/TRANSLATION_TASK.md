# Indic caption translation — parallel task brief

**Deliverable:** English motion captions translated to **Hindi, Bengali, Tamil, Telugu**, split
three ways so three machines can run at once and the results merge cleanly.

Everything you need is this document plus `translate_captions.py`. If you are an agent running
this for someone: read the **Rules** section before you write any code — the constraints there
exist because breaking them produces damage that is invisible until much later.

---

## 1. Context — what this feeds

We drive a **frozen** humanoid-motion model (OMG, Unitree G1) from Indic-language text. A small
adapter (4.7M trainable parameters; the 100M-parameter motion model, MuRIL and t5 are all
frozen) learns to map Indic captions into the text-embedding space the frozen model already
understands.

A previous run trained on **Hindi only** and reached +10.46% caption sensitivity against a
+1.80% baseline for the stock model given Hindi. The next run scales to **four languages at
once with a single shared adapter**, which is why we need all four translations of every
caption.

**Your translations become the training signal.** A caption that is wrong, empty, or silently
left in English becomes a training pair that teaches the adapter nothing, or worse, teaches it
the wrong mapping. Volume is not the goal; completeness and correctness are.

---

## 2. The task in one sentence

Take your assigned third of `manifest.csv`, translate the `caption_1` column into four
languages, verify it, and hand back one CSV.

| | |
|---|---|
| total captions | ~44,854 |
| languages | hi, bn, ta, te |
| total cells | ~179,400 |
| **per person** | **~14,950 rows / ~59,800 cells** |
| measured throughput | ~0.5 cells/s, ~65% first-pass success (see below) |

The job is **resumable**. Stop it, close the laptop, re-run the same command — it picks up
where it left off. You do not need to finish in one sitting.

### This takes days, not hours. Use `--loop` and leave it running.

Measured on 2026-09-07 against the live endpoint:

| spacing | success rate | throughput |
|---|---|---|
| 0.1 s | 45% | — |
| 0.5 s | 65% | 0.49 cells/s |
| 1.5 s | 70% | 0.33 cells/s |

The free endpoint throttles hard, and **slowing down barely helps** — there is a ~30% failure
floor that is not rate-driven. At ~0.5 cells/s a single pass over ~60,000 cells is around
**34 hours**, and roughly a third of the cells will need another pass.

So: **run it with `--loop` and leave it alone.** The script makes repeated passes, filling
only the gaps each time, cooling down between them. It stops when every cell is filled, or
when a pass makes no progress at all — which means the endpoint is refusing rather than
failing, and the fix is to wait rather than retry.

Expect **2-4 days** of wall clock. It checkpoints every 100 rows, so interruptions,
reboots and closed laptops cost nothing: re-run the identical command and it resumes.

---

## 3. Setup

```bash
pip install deep-translator pandas
```

You need a working internet connection. No API key, no GPU, no account.

Files you should have:

```
manifest.csv            <- the input, sent to you (do not edit it)
translate_captions.py   <- the script
```

---

## 4. Run your shard

Shards are assigned by row position, `index % 3`, so the three of us cover every row exactly
once with no coordination:

| person | command |
|---|---|
| 1 | `python translate_captions.py --shard 0 --shards 3 --in manifest.csv --loop` |
| 2 | `python translate_captions.py --shard 1 --shards 3 --in manifest.csv --loop` |
| 3 | `python translate_captions.py --shard 2 --shards 3 --in manifest.csv --loop` |

**Smoke test first** — 20 rows, about a minute, confirms your network and install before you
commit to hours:

```bash
python translate_captions.py --shard 0 --shards 3 --in manifest.csv --limit 20
```

Open the output and confirm the four new columns contain non-Latin script. Then delete
`translations_shard0.csv` and `translations_shard0.cache.json` and start the real run.

Progress prints every 100 rows with a rate and ETA.

### Do NOT use `--batch`

Measured 2026-09-07: `deep_translator.translate_batch()` is **not** a batch API. Its
implementation (`deep_translator/base.py:181`) is a Python loop calling `translate()` once per
string. It issues exactly the same number of requests, with no delay between them and no
retry, so under throttling it is strictly **worse** than the default path and a single failure
aborts the whole batch. The flag exists only for historical reasons. Leave it off.

---

## 5. Verify before handing back

**This step is not optional.** Run it and paste the output when you hand the file over:

```bash
python translate_captions.py --verify --in manifest.csv --out translations_shard0.csv --shard 0 --shards 3
```

It checks row count, that every `clip_id` in your shard is present, that none belong to another
shard, that there are no duplicates, and that all four language columns are fully populated. It
prints `PASS` or `FAIL`.

It also reports, per language:

- **`filled`** — must equal your row count. Anything less means cells failed.
- **`untranslated(==en)`** — rows where output is byte-identical to the English. A handful is
  normal (numbers, proper nouns). Hundreds means something is wrong; say so rather than
  shipping it.
- **`distinct`** — expected to be *lower* than `filled`. Machine translation collapses
  different English sentences onto the same Indic string (we measured *"stop walking forward"*
  and *"stop jogging forward"* both mapping to one Hindi sentence). That is expected and we
  handle it downstream — just report the number.

If it says FAIL, re-run the translate command. It resumes and fills only the gaps.

---

## 6. What to hand back

Two files:

```
translations_shard<K>.csv          <- the result
translations_shard<K>.cache.json   <- optional but useful; lets us re-check without re-calling
```

Plus the `--verify` output pasted as text.

The CSV has exactly these columns:

```
clip_id, caption_1, caption_1_hi, caption_1_bn, caption_1_ta, caption_1_te
```

---

## 7. Rules

These are ordered by how much damage breaking them causes.

1. **Never modify `manifest.csv`.** The merge joins on it. If your copy differs from ours, rows
   silently mismatch. The script never writes to it; do not do so by hand either.

2. **Always UTF-8.** Every read and write in the script pins `encoding="utf-8"`. On Windows the
   default is cp1252, which cannot represent any of these four scripts. It fails loudly on
   write, but *silently mangles* on read — so if you open the CSV in Excel and re-save it, you
   may destroy every translation without any error appearing. **Do not open the output in
   Excel.** If you must look at it, use a text editor set to UTF-8, or `pandas.read_csv(...,
   encoding="utf-8")`.

3. **Do not reorder or drop rows**, and do not sort the CSV. The merge is by `clip_id`, but a
   reordered file makes every mismatch harder to diagnose.

4. **Do not change `--shard` mid-run.** The output file is tied to one shard. If you need to
   switch, use a different `--out` path.

5. **Do not fill blanks by hand.** If cells cannot be translated, leave them empty and tell us
   the count. An empty cell is a known gap we can re-run; a hand-written guess is an unknown
   error that reaches training.

6. **Report anything odd rather than fixing it quietly.** A caption that produces a strange
   translation is data we want to see.

---

## 8. Troubleshooting

**Repeated `TranslationNotFound` messages.** This is throttling, not bad data. The script
backs off (capped at 8 s so a long tail of retries cannot dominate the run) and moves on.
Re-run the same command afterwards to fill the gaps. If a whole pass makes no progress, wait
~15 minutes and use `--sleep 1.0`.

**`UnicodeEncodeError: 'charmap' codec`.** You should never see this from the script — it
forces UTF-8 on stdout at startup. If you see it from your *own* debugging code, that is the
Windows cp1252 console, not a translation failure: add
`sys.stdout.reconfigure(encoding="utf-8")` before printing any translated text.

**`ModuleNotFoundError: deep_translator`** — `pip install deep-translator` (hyphen in the
package name, underscore in the import).

**Run interrupted.** Just re-run the identical command. Progress is checkpointed every 100
rows and the translation cache is on disk.

**Verify says rows are missing.** Re-run the translate command; it fills only the gaps. If the
same rows keep failing, send us those `clip_id` values.

**Output looks like `????` or boxes.** That is a display/encoding issue in your viewer, not
necessarily in the file. Check with:
```python
import pandas as pd
d = pd.read_csv("translations_shard0.csv", encoding="utf-8")
print(d[["caption_1", "caption_1_ta"]].head())
```

---

## 9. What happens after (so the rules make sense)

We concatenate the three shard CSVs, assert the `clip_id` sets are disjoint and their union
equals the manifest exactly, then join the four language columns back onto `manifest.csv`.

Then two things run that depend on your output being intact:

- **MT-collapse detection.** Where several English captions map onto one Indic string, those
  clips must be grouped so they cannot be split across train and test. This silently broke our
  data splits once already, which is why we measure it rather than assume it.
- **Leakage-safe splitting.** Clips are merged transitively by shared source motion, by any
  shared caption string in any of the five languages, and by mirror relationships — then splits
  are drawn over groups, never over individual clips.

Both depend on your translations being complete and unaltered. That is the whole reason for
the verify step and for rule 1.

---

## 10. Quick reference

```bash
pip install deep-translator pandas

# smoke test (1 min)
python translate_captions.py --shard 0 --shards 3 --in manifest.csv --limit 20

# real run -- start it and leave it; expect 2-4 days
python translate_captions.py --shard 0 --shards 3 --in manifest.csv --loop

# verify, then hand back the CSV + this output
python translate_captions.py --verify --in manifest.csv --out translations_shard0.csv --shard 0 --shards 3
```

Replace `0` with your assigned shard number. Person 1 → 0, Person 2 → 1, Person 3 → 2.
