#!/usr/bin/env python3
"""Translate English motion captions to Hindi, Bengali, Tamil and Telugu.

Same GoogleTranslator path that produced the Run-2 corpus, whose output was manually spot
checked and judged good. Four changes, all of them for running this across three machines:

  1. TAMIL ADDED  (ta) -- Run 2 shipped hi/bn/te only.
  2. SHARDING     --shard K --shards N takes rows where index %% N == K, so three people
                   cover a disjoint partition with no coordination and no overlap.
  3. SEPARATE OUTPUT  the input CSV is never modified. Each shard writes its own file, so a
                   crash on one machine cannot corrupt anyone else's work or the source.
  4. PERSISTENT CACHE  survives restarts, so a resumed run does not re-pay for work already
                   done.

DO NOT USE --batch
  Measured 2026-09-07: deep_translator.translate_batch is NOT a batch API. Its implementation
  (deep_translator/base.py:181) loops calling translate() once per string, so it issues the
  same number of requests with no spacing and no retry -- strictly worse under throttling,
  and one failure aborts the whole batch.

ENCODING
  Every read and write pins encoding="utf-8". On Windows the default is cp1252, which cannot
  represent Devanagari, Bengali, Tamil or Telugu -- it fails loudly on write, but a plain
  read of someone else's file will silently mangle it. Never drop the explicit encoding.

USAGE
  python translate_captions.py --shard 0 --shards 3 --in manifest.csv
  python translate_captions.py --verify --in manifest.csv --out translations_shard0.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which cannot encode Devanagari, Bengali,
# Tamil or Telugu. Printing a translated string then raises UnicodeEncodeError --
# which looks exactly like a translation failure and is not one. Measured
# 2026-09-07 while smoke testing this script. Force UTF-8 so it cannot occur.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

TARGETS = {
    "hi": "caption_1_hi",     # Hindi     (Devanagari)
    "bn": "caption_1_bn",     # Bengali   (Bengali)
    "ta": "caption_1_ta",     # Tamil     (Tamil)      <- added for Run 3
    "te": "caption_1_te",     # Telugu    (Telugu)
}
CHECKPOINT_EVERY = 100
# Measured 2026-09-07: at 0.1 s between calls, 11 of 20 cells failed with
# TranslationNotFound -- the free endpoint throttles back-to-back requests.
# A slow run that finishes beats a fast one that silently drops half its cells.
SLEEP_BETWEEN = 0.4
MAX_RETRIES = 4


def is_filled(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def translate_one(text: str, lang: str, cache: dict) -> str | None:
    """Same call, same retry shape as the verified Run-2 script; harder backoff on throttling."""
    from deep_translator import GoogleTranslator

    key = "%s	%s" % (lang, text)
    if key in cache:
        return cache[key]
    for attempt in range(MAX_RETRIES):
        try:
            out = GoogleTranslator(source="en", target=lang).translate(text)
            if out is None or str(out).strip() == "":
                raise ValueError("empty translation returned")
            cache[key] = out
            return out
        except Exception as e:                                        # noqa: BLE001
            # TranslationNotFound is throttling, not bad input: the same text usually
            # succeeds once the endpoint stops rate limiting, so back off harder for it.
            # Throttling is bursty, not persistent: waiting 48 s does not help, and at
            # ~60k cells per shard a long tail of maximal backoffs dominates the runtime.
            # Fail fast and let the resume pass pick these up instead -- measured 2026-09-07,
            # 0.4 s spacing gives ~85% first-pass success, so 2-3 passes converge.
            throttled = "TranslationNotFound" in type(e).__name__
            wait = min(2 ** attempt, 8) if throttled else (2 ** attempt)
            print("  retry %d/%d %s after %ds (%s)"
                  % (attempt + 1, MAX_RETRIES, lang, wait, type(e).__name__), flush=True)
            time.sleep(wait)
    return None


def translate_batch(texts: list[str], lang: str, cache: dict) -> list[str | None]:
    from deep_translator import GoogleTranslator

    todo = [t for t in texts if "%s\t%s" % (lang, t) not in cache]
    if todo:
        for attempt in range(MAX_RETRIES):
            try:
                got = GoogleTranslator(source="en", target=lang).translate_batch(todo)
                for t, o in zip(todo, got):
                    cache["%s\t%s" % (lang, t)] = o
                break
            except Exception as e:                                    # noqa: BLE001
                wait = 2 ** attempt
                print("  batch retry %d/%d for %s after %ds: %s"
                      % (attempt + 1, MAX_RETRIES, lang, wait, str(e)[:90]), flush=True)
                time.sleep(wait)
    return [cache.get("%s\t%s" % (lang, t)) for t in texts]


def verify(src: Path, out: Path, shard: int, shards: int) -> int:
    """Check a finished shard before it is handed back. Fails loudly, never silently."""
    df = pd.read_csv(src, low_memory=False, encoding="utf-8")
    mine = df[df.index % shards == shard]
    got = pd.read_csv(out, low_memory=False, encoding="utf-8")
    print("shard %d/%d" % (shard, shards))
    print("  expected rows : %d" % len(mine))
    print("  rows present  : %d" % len(got))
    ok = True
    if len(got) != len(mine):
        print("  !! ROW COUNT MISMATCH"); ok = False
    missing = set(mine["clip_id"]) - set(got["clip_id"])
    extra = set(got["clip_id"]) - set(mine["clip_id"])
    if missing:
        print("  !! %d clip_id missing (e.g. %s)" % (len(missing), list(missing)[:5])); ok = False
    if extra:
        print("  !! %d clip_id not in this shard (e.g. %s)" % (len(extra), list(extra)[:5])); ok = False
    if got["clip_id"].duplicated().any():
        print("  !! duplicate clip_id"); ok = False
    for lang, col in TARGETS.items():
        if col not in got.columns:
            print("  !! missing column %s" % col); ok = False; continue
        n_ok = int(got[col].apply(is_filled).sum())
        same_as_en = int((got[col].astype(str).str.strip()
                          == got["caption_1"].astype(str).str.strip()).sum())
        uniq = got[col][got[col].apply(is_filled)].nunique()
        print("    %-3s filled %6d/%6d  untranslated(==en) %4d  distinct %6d"
              % (lang, n_ok, len(got), same_as_en, uniq))
        if n_ok < len(got):
            ok = False
    print("\n  VERDICT: %s" % ("PASS - ready to hand back" if ok else "FAIL - do not hand back"))
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="manifest.csv")
    ap.add_argument("--out", default=None, help="default translations_shard<K>.csv")
    ap.add_argument("--source-col", default="caption_1")
    ap.add_argument("--id-col", default="clip_id")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--batch", action="store_true",
                    help="DO NOT USE: deep_translator.translate_batch is a loop over "
                         "translate(), not a batch API -- same request count, no retry")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N rows")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN,
                    help="seconds between calls; raise on TranslationNotFound")
    ap.add_argument("--abort-after", type=int, default=60,
                    help="stop if this many cells fail in a row with none "
                         "succeeding -- means blocked, not throttled")
    ap.add_argument("--loop", action="store_true",
                    help="keep making passes until every cell is filled")
    ap.add_argument("--cooldown", type=int, default=300,
                    help="seconds between passes in --loop mode")
    ap.add_argument("--max-passes", type=int, default=40)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out) if a.out else Path("translations_shard%d.csv" % a.shard)
    if a.verify:
        return verify(src, out, a.shard, a.shards)

    if not 0 <= a.shard < a.shards:
        raise SystemExit("--shard must be in [0, %d)" % a.shards)

    df = pd.read_csv(src, low_memory=False, encoding="utf-8")
    if a.source_col not in df.columns:
        raise SystemExit("no column %r in %s" % (a.source_col, src))
    if a.id_col not in df.columns:
        raise SystemExit("no column %r in %s -- the merge needs a stable key" % (a.id_col, src))

    # deterministic partition: no coordination needed, and provably disjoint
    mine = df[df.index % a.shards == a.shard].copy()
    if a.limit:
        mine = mine.head(a.limit)
    keep = [a.id_col, a.source_col]
    work = mine[keep].reset_index(drop=True)

    # resume from a previous run of THIS shard
    if out.exists():
        prev = pd.read_csv(out, low_memory=False, encoding="utf-8")
        work = work.merge(prev.drop(columns=[a.source_col], errors="ignore"),
                          on=a.id_col, how="left")
        print("resuming from %s (%d rows already present)" % (out, len(prev)))
    for col in TARGETS.values():
        if col not in work.columns:
            work[col] = pd.NA

    cache_path = out.with_suffix(".cache.json")
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print("cache: %d entries" % len(cache))

    pending = sum(int((~work[c].apply(is_filled)).sum()) for c in TARGETS.values())
    print("shard %d/%d: %d rows, %d translation cells pending (%s)"
          % (a.shard, a.shards, len(work), pending, ", ".join(TARGETS)))
    if pending == 0:
        print("nothing to do"); return 0
    t0, done, since_save, failed = time.time(), 0, 0, 0
    consecutive_fail = 0
    globals()["SLEEP_BETWEEN"] = a.sleep

    def save():
        work.to_csv(out, index=False, encoding="utf-8")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    if a.batch:
        print("BATCH MODE - faster, but not the manually verified path")
        for lang, col in TARGETS.items():
            todo = work.index[~work[col].apply(is_filled)].tolist()
            for s in range(0, len(todo), a.batch_size):
                idx = todo[s:s + a.batch_size]
                texts = [str(work.at[i, a.source_col]) for i in idx]
                for i, o in zip(idx, translate_batch(texts, lang, cache)):
                    if o is not None:
                        work.at[i, col] = o
                done += len(idx)
                if s // max(a.batch_size, 1) % 10 == 0:
                    el = time.time() - t0
                    print("  %s %d/%d  %.0f cells/min  ETA %.1f min"
                          % (lang, s + len(idx), len(todo), 60 * done / max(el, 1),
                             (pending - done) / max(done / max(el, 1), 1e-9) / 60), flush=True)
                    save()
        save()
    else:
        # AUTO-LOOP. Measured 2026-09-07: the free endpoint sustains ~0.5 cells/s at
        # ~65% success, so a 60k-cell shard needs several passes over many hours. Making
        # a person re-run it by hand for days invites them to stop at 85% and call it
        # done. This repeats passes until nothing is left, or until a pass makes no
        # progress at all (which means waiting, not retrying, is the fix).
        max_passes = a.max_passes if a.loop else 1
        for _pass in range(max_passes):
            before = sum(int((~work[c].apply(is_filled)).sum()) for c in TARGETS.values())
            for i in work.index:
                text = work.at[i, a.source_col]
                if not is_filled(text):
                    continue
                text = str(text)
                touched = False
                for lang, col in TARGETS.items():
                    if is_filled(work.at[i, col]):
                        continue
                    o = translate_one(text, lang, cache)
                    if o is not None:
                        work.at[i, col] = o
                        touched = True
                        done += 1
                        consecutive_fail = 0
                    else:
                        failed += 1
                        consecutive_fail += 1
                        # ABORT EARLY WHEN THE ENDPOINT IS BLOCKING OUTRIGHT. Measured
                        # 2026-09-07: a fully blocked IP fails every cell, and each failed
                        # cell costs ~15 s of backoff. One 15k-row pass would grind for
                        # ~250 h before the end-of-pass no-progress check could ever fire.
                        # Nothing succeeding means wait, not retry.
                        if consecutive_fail >= a.abort_after:
                            print("")
                            print("  ABORTING: %d cells failed in a row, none succeeded."
                                  % consecutive_fail)
                            print("  This IP is blocked outright, not throttled.")
                            print("  Wait a few hours and re-run, or use another network.")
                            save()
                            return 3
                    time.sleep(SLEEP_BETWEEN)
                if touched:
                    since_save += 1
                    if since_save >= CHECKPOINT_EVERY:
                        save(); since_save = 0
                        el = time.time() - t0
                        rate = done / max(el, 1)
                        print("[checkpoint] row %d/%d  %.0f cells/min  ETA %.1f min"
                              % (i + 1, len(work), 60 * rate,
                                 (pending - done) / max(rate, 1e-9) / 60), flush=True)
            save()
            after = sum(int((~work[c].apply(is_filled)).sum()) for c in TARGETS.values())
            print("[pass %d] %d -> %d cells remaining (%d filled this pass)"
                  % (_pass + 1, before, after, before - after), flush=True)
            if after == 0:
                print("  all cells filled"); break
            if not a.loop:
                break
            if after == before:
                print("  no progress this pass -- the endpoint is refusing, not failing.")
                print("  stopping. wait ~15 min and re-run, or raise --sleep."); break
            print("  cooling down %ds before the next pass ..." % a.cooldown, flush=True)
            time.sleep(a.cooldown)

    print("\nwrote %s  (%d cells translated in %.1f min)"
          % (out, done, (time.time() - t0) / 60))
    still = sum(int((~work[c].apply(is_filled)).sum()) for c in TARGETS.values())
    if still:
        print("  !! %d cells still empty (%d gave up after %d retries)."
              % (still, failed, MAX_RETRIES))
        print("     Re-run the SAME command -- it resumes and fills only the gaps.")
        print("     If it keeps failing, raise --sleep (e.g. --sleep 1.0).")
    print("now run:  python translate_captions.py --verify --in %s --out %s --shard %d --shards %d"
          % (src, out, a.shard, a.shards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
