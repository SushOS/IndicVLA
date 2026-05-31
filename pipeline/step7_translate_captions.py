"""
translate_captions.py
---------------------
Translates caption_1 through caption_4 into Hindi (hi), Bengali (bn),
and Tamil (ta) for every row in the robot dataset.

New columns added:
  caption_1_hi  caption_1_bn  caption_1_ta
  caption_2_hi  caption_2_bn  caption_2_ta
  caption_3_hi  caption_3_bn  caption_3_ta
  caption_4_hi  caption_4_bn  caption_4_ta

Resumable: re-running skips already-translated cells.
Checkpoints to the same CSV every CHECKPOINT_EVERY rows.
Ctrl-C saves progress before exit.

Usage:
    python translate_captions.py
    python translate_captions.py --csv D:/HumanML3d/g1_dataset_robot_v2.csv
"""

import argparse
import signal
import sys
import time

import pandas as pd

try:
    from deep_translator import GoogleTranslator
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "deep-translator", "-q"],
                  check=True)
    from deep_translator import GoogleTranslator

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CSV      = "D:/HumanML3d/g1_dataset_robot_v2.csv"
SOURCE_COLS      = ["caption_1", "caption_2", "caption_3", "caption_4"]
LANGUAGES        = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu"}
CHECKPOINT_EVERY = 50    # save to disk every N rows that had work done
SLEEP_BETWEEN    = 0.15  # seconds between API calls
MAX_RETRIES      = 3     # attempts per cell before giving up

# Derived: caption_1 -> {hi: caption_1_hi, bn: caption_1_bn, ta: caption_1_ta}
COL_MAP = {
    src: {lang: f"{src}_{lang}" for lang in LANGUAGES}
    for src in SOURCE_COLS
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_filled(val) -> bool:
    return pd.notna(val) and str(val).strip() != ""


def translate_one(text: str, lang: str, cache: dict):
    key = (lang, text)
    if key in cache:
        return cache[key]
    for attempt in range(MAX_RETRIES):
        try:
            result = GoogleTranslator(source="en", target=lang).translate(text)
            if result:
                cache[key] = result
                return result
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] lang={lang}  wait={wait}s  {exc}")
            time.sleep(wait)
    # All retries exhausted — mark as failed so we don't retry forever this run
    cache[key] = None
    return None


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for src_col, lang_cols in COL_MAP.items():
        for tgt_col in lang_cols.values():
            if tgt_col not in df.columns:
                df[tgt_col] = pd.NA
    return df


def count_pending(df: pd.DataFrame) -> int:
    total = 0
    for src_col, lang_cols in COL_MAP.items():
        src_filled = df[src_col].apply(is_filled)
        for tgt_col in lang_cols.values():
            total += (src_filled & ~df[tgt_col].apply(is_filled)).sum()
    return int(total)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(csv_path: str):
    df = load_df(csv_path)
    n  = len(df)
    cache: dict = {}

    pending = count_pending(df)
    print(f"CSV        : {csv_path}")
    print(f"Rows       : {n:,}")
    print(f"Languages  : {', '.join(f'{v} ({k})' for k, v in LANGUAGES.items())}")
    print(f"Columns    : {', '.join(SOURCE_COLS)}")
    print(f"Pending    : {pending:,} translation cells")
    print(f"New columns: {', '.join(f'{s}_{l}' for s in SOURCE_COLS for l in LANGUAGES)}")
    if pending == 0:
        print("All cells already translated — nothing to do.")
        return
    # Rough ETA: ~0.3s per translation (sleep + API)
    eta_h = pending * (SLEEP_BETWEEN + 0.15) / 3600
    print(f"Est. time  : ~{eta_h:.1f} hours (resumable at any point)\n")

    # ── Graceful Ctrl-C ───────────────────────────────────────────────────────
    def _on_interrupt(sig, frame):
        print("\n[interrupt] Saving checkpoint before exit ...")
        df.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path}")
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_interrupt)

    # ── Translation loop ──────────────────────────────────────────────────────
    since_save   = 0
    translations = 0
    errors       = 0
    t0           = time.time()

    for idx, row in df.iterrows():
        row_did_work = False

        for src_col, lang_cols in COL_MAP.items():
            src_text = row[src_col]
            if not is_filled(src_text):
                continue
            src_text = str(src_text)

            for lang, tgt_col in lang_cols.items():
                if is_filled(df.at[idx, tgt_col]):
                    continue                          # already translated

                result = translate_one(src_text, lang, cache)
                if result is not None:
                    df.at[idx, tgt_col] = result
                    translations += 1
                    row_did_work  = True
                else:
                    errors += 1

                time.sleep(SLEEP_BETWEEN)

        if row_did_work:
            since_save += 1

        # ── Checkpoint ────────────────────────────────────────────────────────
        if since_save >= CHECKPOINT_EVERY:
            df.to_csv(csv_path, index=False)
            since_save = 0
            elapsed  = time.time() - t0
            rate     = translations / elapsed if elapsed > 0 else 0
            remaining = count_pending(df)
            eta_min  = (remaining / rate / 60) if rate > 0 else float("inf")
            print(
                f"[checkpoint] row {idx + 1:>6,}/{n:,}  "
                f"done={translations:,}  errors={errors}  "
                f"rate={rate:.1f}/s  ETA≈{eta_min:.0f} min"
            )

        # ── Progress ticker ───────────────────────────────────────────────────
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = translations / elapsed if elapsed > 0 else 0
            print(f"[progress]   row {idx + 1:>6,}/{n:,}  translated={translations:,}  {rate:.1f} trans/s")

    # ── Final save ────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    elapsed = time.time() - t0

    print()
    print("=" * 55)
    print("Done!")
    print(f"  Rows        : {n:,}")
    print(f"  Translated  : {translations:,}")
    print(f"  Errors      : {errors}")
    print(f"  Time        : {elapsed / 60:.1f} min")
    print(f"  Output      : {csv_path}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate captions to Hindi, Bengali, Tamil")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV file (overwritten in place)")
    args = parser.parse_args()
    main(args.csv)
