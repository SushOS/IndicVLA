#!/usr/bin/env python3
"""3-arm comparable benchmark: English / Hindi-via-adapter / MT-pivot, on OUR test split.

WHY NOT THE OFFICIAL 1024-SAMPLE OMG-DATA MANIFEST
----------------------------------------------------
`benchmark artifact`/`benchmark text` route sample resolution through the LeRobot dataset
object, which demands globally-contiguous parquet indices from 0 -- the same constraint
that forced a 62 GB / ~90 minute download for EXP-F. Re-paying that cost only to borrow a
gallery drawn from 12 unrelated corpora would be wasted spend: EXP-F already proved this
project's harness reproduces OMG's published number EXACTLY (REFERENCE R@1 = 0.6680, all
printed digits), so that calibration is banked and does not need re-proving.

Instead: reuse OMG's own evaluator METRIC FUNCTIONS (motion_fid, motion_kid, diversity,
_text_retrieval_summary -- same code EXP-F used) but apply them to OUR test split, which
already has real G1 motion + real English + real Hindi captions (2.3 GB, no LeRobot
dependency). All three arms score against the SAME gallery, so the comparison is internally
apples-to-apples even though the absolute numbers won't match OMG's published 1024-sample
table (different gallery composition, same as any train/test-split choice would produce).

THE THREE ARMS
--------------
  A  ENGLISH        frozen t5-base + the clip's real English caption      (ceiling)
  B  HINDI-ADAPTER   frozen MuRIL + trained adapter + the clip's real Hindi caption
  C  MT-PIVOT        Hindi caption -> IndicTrans2 hi->en -> frozen t5-base
                      (tests "why not just translate at inference" directly)

All three generate from the SAME seed poses with the SAME noise seed. Retrieval always
scores against the REAL English caption (dataset ground truth), matching the artifact
runner's own policy (research-log 94: captions for retrieval are re-read from the dataset,
never from what drove generation) -- this is what makes a Hindi-input arm scoreable at all
against a T5-3B evaluator that cannot itself read Devanagari.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")
from maila_train import WindowDataset, build  # noqa: E402


def load_mt(model_dir: str, device: torch.device):
    """IndicTrans2 hi->en if available, else NLLB-600M. Both use the same call contract here."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    is_it2 = "indictrans2" in model_dir.lower()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=is_it2)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=is_it2).to(device).eval()
    return tok, model, is_it2


@torch.no_grad()
def mt_translate(tok, model, is_it2, texts, device, batch_size=32):
    out = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        if is_it2:
            from IndicTransToolkit.processor import IndicProcessor
            ip = IndicProcessor(inference=True)
            prepped = ip.preprocess_batch(chunk, src_lang="hin_Deva", tgt_lang="eng_Latn")
            enc = tok(prepped, return_tensors="pt", padding=True, truncation=True).to(device)
            # use_cache=False: IndicTrans2's custom remote-code forward() assumes an older
            # transformers past_key_values init shape and crashes on 4.57.6 with
            # AttributeError: 'NoneType' object has no attribute 'shape'. Verified 2026-08-22:
            # disabling the KV cache sidesteps it (captions are short, no real cost) and
            # round-trips correctly, e.g. "...steps to their right" -> "The person steps to
            # their right." Fixing the remote code itself is out of scope.
            gen = model.generate(**enc, max_length=128, num_beams=4, use_cache=False)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            out.extend(ip.postprocess_batch(dec, lang="eng_Latn"))
        else:
            tok.src_lang = "hin_Deva"
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                                 max_length=128, num_beams=4)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


@torch.no_grad()
def generate_batch(model, ds, idx, captions, seed, num_frames, cfg_scale, dev, batch_size=32):
    """Generate qpos for a list of window indices under given captions, fixed noise seed."""
    qpos_all = []
    for s in range(0, len(idx), batch_size):
        sel = idx[s:s + batch_size]
        caps = [captions[j] for j in range(s, min(s + batch_size, len(idx)))]
        w = torch.from_numpy(ds.x[sel]).to(dev)
        L = ds.L
        ap_ = np.stack([np.asarray(ds.man.iloc[i]["anchor_root_pos"], np.float32) for i in sel])
        aq_ = np.stack([np.asarray(ds.man.iloc[i]["anchor_root_quat"], np.float32) for i in sel])
        b = {
            "history_features": w[:, :L], "prev_state_features": w[:, :L],
            "canon_root_pos": torch.from_numpy(ap_).to(dev),
            "canon_root_quat": torch.from_numpy(aq_).to(dev),
            "caption": caps, "has_text": torch.ones(len(sel), dtype=torch.bool, device=dev),
            "fps": torch.full((len(sel),), 30.0, device=dev),
        }
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        g = model.generate(b, num_frames=num_frames, cfg_scale=cfg_scale)
        qpos_all.append(g["qpos_36"].float().cpu())
    return torch.cat(qpos_all)


def score_arm(label, qpos, real_en_captions, ds_names, evaluator, text_enc, kinematics,
             representation, dev, out_dir):
    from omg.benchmarks.metrics import diversity, motion_fid, motion_kid
    from omg.benchmarks.runners.common import (
        _encode_motion_embeddings, _motion_for_evaluator,
    )
    from omg.benchmarks.runners.text import _text_retrieval_summary, TEXT_RETRIEVAL_BATCH_SIZE

    n = qpos.shape[0]
    valid = torch.ones(n, qpos.shape[1], dtype=torch.bool)
    motion_key = "body_pos_local"
    motion = _motion_for_evaluator(qpos, motion_key, kinematics=kinematics)
    m_emb = _encode_motion_embeddings(evaluator, motion, valid, batch_size=32, device=dev)

    with torch.inference_mode():
        t_chunks = []
        for s in range(0, len(real_en_captions), 32):
            enc = text_enc(real_en_captions[s:s + 32], device=dev)
            t_chunks.append(enc.detach().cpu())
        t_emb = torch.cat(t_chunks).numpy()

    n_use = (n // TEXT_RETRIEVAL_BATCH_SIZE) * TEXT_RETRIEVAL_BATCH_SIZE
    if n_use < TEXT_RETRIEVAL_BATCH_SIZE:
        raise SystemExit(f"{label}: only {n} samples, need >= {TEXT_RETRIEVAL_BATCH_SIZE}")
    m_emb_u, t_emb_u, names_u = m_emb[:n_use], t_emb[:n_use], ds_names[:n_use]

    retrieval = _text_retrieval_summary(m_emb_u, t_emb_u, dataset_names=names_u)
    div = float(diversity(m_emb_u))
    # FID/KID need REFERENCE embeddings, computed once and shared across arms -> caller adds them

    result = {
        "label": label, "n": int(n_use),
        "r_precision": retrieval["r_precision"], "matching_score": retrieval["matching_score"],
        "mean_text_rank": retrieval["mean_text_rank"], "median_text_rank": retrieval["median_text_rank"],
        "diversity": div,
    }
    (out_dir / f"{label}_result.json").write_text(json.dumps(result, indent=2))
    np.save(out_dir / f"{label}_motion_emb.npy", m_emb_u)
    print(f"  {label:16s} R@1={retrieval['r_precision'][0]:.4f} R@2={retrieval['r_precision'][1]:.4f} "
          f"R@3={retrieval['r_precision'][2]:.4f} match={retrieval['matching_score']:.4f} "
          f"medR={retrieval['median_text_rank']:.1f}", flush=True)
    return result, m_emb_u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/omg125")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter-ckpt", required=True)
    ap.add_argument("--omg", default="/workspace/models/omg/checkpoints/updated/100m/sstep=170000.ckpt")
    ap.add_argument("--evaluator", default="/workspace/models/omg/evaluator/step_004000.pt")
    ap.add_argument("--t5", default="/workspace/models/t5-base-local")
    ap.add_argument("--t5-3b", default="/workspace/models/t5-3b-local")
    ap.add_argument("--muril", default="/workspace/models/muril-base-cased")
    ap.add_argument("--mt-model", default="/workspace/models/indictrans2-indic-en")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/out/three_arm")
    ap.add_argument("--pivot-only", action="store_true",
                    help="reuse saved REFERENCE/ENGLISH/HINDI_ADAPTER embeddings+results "
                         "(from a prior run with the SAME --seed/--n/--split, so idx/en/hi "
                         "reproduce identically) and only run arm C + the final summary. "
                         "For resuming after an MT-model-only crash without re-paying GPU "
                         "time on the two arms that already succeeded.")
    a = ap.parse_args()

    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = WindowDataset(a.data, a.split, "hi")
    rng = np.random.default_rng(a.seed)
    from omg.benchmarks.runners.text import TEXT_RETRIEVAL_BATCH_SIZE
    n_use = min(a.n, (len(ds) // TEXT_RETRIEVAL_BATCH_SIZE) * TEXT_RETRIEVAL_BATCH_SIZE)
    idx = rng.choice(len(ds), size=n_use, replace=False)
    en = [ds.en[i] for i in idx]
    hi = [ds.hi[i] for i in idx]
    ds_names = [str(ds.man.iloc[i].get("category", "test"))[:24] for i in idx]
    print(f"{n_use} test windows, {len(set(en))} distinct English captions")

    from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
    from omg.benchmarks.runners.common import _load_evaluator_motion_encoder, _motion_for_evaluator
    from omg.benchmarks.evaluator.text_encoder import TextEncoder as EvalTextEncoder

    if a.pivot_only:
        print("--pivot-only: reusing saved REFERENCE/ENGLISH/HINDI_ADAPTER, running arm C only")
        ref_result = json.loads((out / "REFERENCE_result.json").read_text())
        res_en = json.loads((out / "ENGLISH_result.json").read_text())
        res_hi = json.loads((out / "HINDI_ADAPTER_result.json").read_text())
        ref_emb = np.load(out / "REFERENCE_motion_emb.npy")
        emb_en = np.load(out / "ENGLISH_motion_emb.npy")
        emb_hi = np.load(out / "HINDI_ADAPTER_motion_emb.npy")
        evaluator, ckpt = _load_evaluator_motion_encoder(a.evaluator, "body_pos_local", dev)
        text_enc = EvalTextEncoder(output_dim=512, model_name="t5-3b").to(dev).eval()
        text_enc.proj.load_state_dict(ckpt["text_proj_state_dict"], strict=True)
        for p in text_enc.parameters():
            p.requires_grad_(False)
        from maila_train import build as build_model
        from omg.generation.conditions.t5 import FrozenT5TextEncoder  # noqa: F401
        rep_model = build_model("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
        representation, kinematics = rep_model.representation, rep_model.representation.kinematics
        del rep_model
        torch.cuda.empty_cache()
        print(f"  loaded: REFERENCE n={ref_result['n']}  ENGLISH n={res_en['n']}  "
             f"HINDI_ADAPTER n={res_hi['n']}")
        skip_to_pivot = True
    else:
        skip_to_pivot = False

    if not a.pivot_only:
        print("\n-- REFERENCE gate: real motion vs real English caption --")
        evaluator, ckpt = _load_evaluator_motion_encoder(a.evaluator, "body_pos_local", dev)
        # Verified against the real checkpoint (2026-08-22): top-level keys are
        # ['motion_encoder', 'text_proj_state_dict', 'optimizer', 'logit_scale', 'step',
        # 'config']. There is no 'text_encoder' key -- only the trained proj layer is stored
        # (Linear+LayerNorm, 4 tensors); the frozen t5-3b body is loaded fresh from disk.
        # Loading with any other key silently leaves `proj` at its RANDOM init, which would
        # make every retrieval number in this benchmark meaningless while looking normal.
        if "text_proj_state_dict" not in ckpt:
            raise SystemExit(f"evaluator checkpoint missing 'text_proj_state_dict'; "
                             f"has keys: {list(ckpt.keys())}")
        text_enc = EvalTextEncoder(output_dim=512, model_name="t5-3b").to(dev).eval()
        text_enc.proj.load_state_dict(ckpt["text_proj_state_dict"], strict=True)
        for p in text_enc.parameters():
            p.requires_grad_(False)

        from maila_train import build as build_model
        rep_model = build_model("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
        representation, kinematics = rep_model.representation, rep_model.representation.kinematics
        real_qpos = torch.from_numpy(ds.x[idx][:, ds.L:]).float()   # (N,60,125)->wrong, need qpos not features
        # NOTE: ds.x stores CANONICALISED 125-D features, not raw qpos_36. Real reference motion
        # for the evaluator gate must come from decoding, not from re-treating features as qpos.
        from omg.motion.feature_codec import MotionComponents
        comps = representation.codec.split_features(torch.from_numpy(ds.x[idx][:, ds.L:]).to(dev))
        ap_ = torch.from_numpy(np.stack([np.asarray(ds.man.iloc[i]["anchor_root_pos"], np.float32) for i in idx])).to(dev)
        aq_ = torch.from_numpy(np.stack([np.asarray(ds.man.iloc[i]["anchor_root_quat"], np.float32) for i in idx])).to(dev)
        real_qpos = representation.codec.decode_to_world_qpos36(comps, ap_, aq_).float().cpu()

        ref_result, ref_emb = score_arm("REFERENCE", real_qpos, en, ds_names, evaluator, text_enc,
                                        kinematics, representation, dev, out)
        print(f"  (checkpoint-independent Gate A analogue on OUR gallery; not directly 0.6680 -- "
             f"different gallery composition, see module docstring)")

        print("\n-- ARM A: ENGLISH (frozen t5) --")
        m_en = build_model("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
        from omg.generation.conditions.t5 import FrozenT5TextEncoder
        m_en.text_encoder._t5 = FrozenT5TextEncoder(model_name=a.t5, max_length=50, output_dim=768).to(dev).eval()
        m_en.text_encoder.passthrough_t5 = True
        q_en = generate_batch(m_en, ds, idx, en, a.seed, a.num_frames, a.cfg_scale, dev)
        res_en, emb_en = score_arm("ENGLISH", q_en, en, ds_names, evaluator, text_enc, kinematics,
                                   representation, dev, out)
        del m_en; torch.cuda.empty_cache()

        print("\n-- ARM B: HINDI-ADAPTER --")
        m_hi = build_model("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
        st = torch.load(a.adapter_ckpt, map_location="cpu", weights_only=False)
        m_hi.text_encoder.adapter.load_state_dict(st["adapter"])
        m_hi.text_encoder.adapter.eval()
        q_hi = generate_batch(m_hi, ds, idx, hi, a.seed, a.num_frames, a.cfg_scale, dev)
        res_hi, emb_hi = score_arm("HINDI_ADAPTER", q_hi, en, ds_names, evaluator, text_enc, kinematics,
                                   representation, dev, out)
        del m_hi; torch.cuda.empty_cache()

    print("\n-- ARM C: MT-PIVOT (hi -> IndicTrans2/NLLB -> en -> t5) --")
    tok, mtm, is_it2 = load_mt(a.mt_model, dev)
    en_pivot = mt_translate(tok, mtm, is_it2, hi, dev)
    del tok, mtm; torch.cuda.empty_cache()
    (out / "mt_pivot_translations.json").write_text(json.dumps(
        [{"hi": h, "en_pivot": p, "en_original": e} for h, p, e in zip(hi, en_pivot, en)][:50],
        indent=2, ensure_ascii=False))
    m_pivot = build_model("100m", a.omg, a.t5, a.muril, dev, True, True, 0.0)
    m_pivot.text_encoder._t5 = FrozenT5TextEncoder(model_name=a.t5, max_length=50, output_dim=768).to(dev).eval()
    m_pivot.text_encoder.passthrough_t5 = True
    q_pivot = generate_batch(m_pivot, ds, idx, en_pivot, a.seed, a.num_frames, a.cfg_scale, dev)
    res_pivot, emb_pivot = score_arm("MT_PIVOT", q_pivot, en, ds_names, evaluator, text_enc, kinematics,
                                     representation, dev, out)
    del m_pivot; torch.cuda.empty_cache()

    print("\n=== FID/KID against REFERENCE embeddings ===")
    from omg.benchmarks.metrics import motion_fid, motion_kid
    for label, emb in (("ENGLISH", emb_en), ("HINDI_ADAPTER", emb_hi), ("MT_PIVOT", emb_pivot)):
        fid = float(motion_fid(ref_emb, emb))
        kid = motion_kid(ref_emb, emb)
        print(f"  {label:16s} FID={fid:.4f}  KID_mean={kid['mean']:.6f}")
        for f in out.glob(f"{label}_result.json"):
            d = json.loads(f.read_text()); d["motion_fid_vs_reference"] = fid; d["motion_kid"] = kid
            f.write_text(json.dumps(d, indent=2))

    print("\n=== SUMMARY TABLE ===")
    print(f"{'arm':16s} {'R@1':>7s} {'R@2':>7s} {'R@3':>7s} {'match':>7s} {'medR':>6s} {'FID':>7s}")
    for label, res in (("ENGLISH", res_en), ("HINDI_ADAPTER", res_hi), ("MT_PIVOT", res_pivot)):
        d = json.loads((out / f"{label}_result.json").read_text())
        r = d["r_precision"]
        print(f"{label:16s} {r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f} {d['matching_score']:7.4f} "
              f"{d['median_text_rank']:6.1f} {d.get('motion_fid_vs_reference',float('nan')):7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
