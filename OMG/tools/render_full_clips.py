#!/usr/bin/env python3
"""Full-length, closed-loop side-by-side renders + the complete comparison matrix.

WHY THIS REPLACES render_panels.py
-----------------------------------
render_panels.py had two problems:

  1. 2-second clips. It rendered one 60-frame training window, far too short to judge
     whether a motion follows its caption.
  2. A BROKEN GROUND TRUTH. It decoded shard features with `rep.decode(ds.x)`. The shards
     hold RAW features (convert_g1_npz_to_omg125.encode_windows ends with
     `codec.assemble_features`, which does not normalise), while `rep.decode` begins with
     `denormalize_features`. That second denormalisation compressed the motion into a
     near-frozen pose -- the "stuck" ground truth.

This file removes that failure mode by construction: **ground truth is read straight from
the source .npz and never touches the codec at all.** base_frame_pos (T,3) +
base_frame_wxyz (T,4) + joint_angles (T,29) == qpos_36 by construction, so GT needs no
encode, no decode and no normalisation. There is nothing left to get the wrong way round.

Everything that must agree with the training data -- fps table, clip slicing, clamping,
resampling, window encoding -- is IMPORTED from convert_g1_npz_to_omg125 rather than
reimplemented, so it cannot drift from what produced the corpus.

CLOSED LOOP
  MotionGenerator.generate(num_frames=N) with N > sequence_length (64) rolls chunk by
  chunk, re-deriving its own history and anchor from its previous output via
  codec.prev_state_features_from_history. Asking for ~270 frames is therefore a genuine
  closed-loop rollout, not four independent 2-second clips.

ALIGNMENT
  Generation is seeded with the clip's first L frames and predicts everything after them,
  so the comparison window is gt_qpos[L : L+N] against each arm's N generated frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/workspace")

BAND = 118
BG = (16, 16, 18)
ARMS_ALL = ["gt", "omg_en", "omg_hi", "maila_en", "maila_hi"]
TITLES = {
    "gt": "GROUND TRUTH (source mocap)",
    "omg_en": "STOCK OMG + English",
    "omg_hi": "STOCK OMG + Hindi",
    "maila_en": "MAILA adapter + English",
    "maila_hi": "MAILA adapter + Hindi",
}
COLOURS = {
    "gt": (150, 220, 165),
    "omg_en": (190, 190, 196),
    "omg_hi": (232, 140, 128),
    "maila_en": (140, 190, 245),
    "maila_hi": (255, 208, 120),
}
LANG_OF = {"gt": None, "omg_en": "en", "omg_hi": "hi", "maila_en": "en", "maila_hi": "hi"}


def draw_header(total_w, cols, fonts):
    """Per-column banner: a coloured rule, the arm name, then that arm own caption.

    Header columns are built from the SAME ordered list used to concatenate the video
    columns, so a label can never drift from the panel it names.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (total_w, BAND), BG)
    d = ImageDraw.Draw(img)
    cw = total_w // len(cols)
    for k, (arm, caption) in enumerate(cols):
        x0 = k * cw
        col = COLOURS[arm]
        d.rectangle([x0 + 10, 8, x0 + cw - 10, 12], fill=col)
        d.text((x0 + 12, 20), TITLES[arm], font=fonts["label"], fill=col)
        if not caption:
            continue
        font = fonts["hi"] if LANG_OF[arm] == "hi" else fonts["en"]
        line, lines = "", []
        for word in str(caption).split():
            trial = (line + " " + word).strip()
            if d.textlength(trial, font=font) > cw - 26 and line:
                lines.append(line)
                line = word
            else:
                line = trial
            if len(lines) == 2:
                break
        if line and len(lines) < 2:
            lines.append(line)
        for j, ln in enumerate(lines[:2]):
            d.text((x0 + 12, 52 + j * 30), ln, font=font, fill=col)
    return np.array(img)


def clip_qpos_from_source(row, conv, cache, token):
    """Full-length clip qpos_36 straight from the source npz. No codec, no normalisation.

    Replicates mode_convert slicing exactly: slice at the SOURCE rate so the clamp is
    measured against real frames, then map into the resampled index space.
    """
    p = conv.fetch_npz(str(row["npz_path"]), cache, None, token)
    if p is None:
        return None, "npz fetch failed"
    q_src = conv.load_qpos36(p)
    if q_src is None:
        return None, "npz unreadable"
    subset = conv.subset_of(str(row["npz_path"]))
    src_fps = conv.SUBSET_FPS.get(subset)
    if src_fps is None:
        return None, "no measured fps for subset " + subset
    T_src = q_src.shape[0]
    q_all = conv.resample_qpos(q_src, src_fps, conv.TARGET_FPS)
    s_src = int(round(float(row["start_s"]) * src_fps))
    e_src = int(round(float(row["end_s"]) * src_fps))
    if s_src >= T_src:
        return None, "start past end of file"
    scale = conv.TARGET_FPS / src_fps
    s0 = int(round(s_src * scale))
    s1 = int(round(min(e_src, T_src) * scale))
    q = q_all[max(s0, 0):min(s1, len(q_all))]
    frac_lost = max(0, e_src - T_src) / max(e_src - s_src, 1)
    return np.ascontiguousarray(q, dtype=np.float32), round(float(frac_lost), 4)


def arm_metrics(gen_q, gt_q, kin, foot_ids, fps, dev):
    """Every arm scored against the SAME ground truth, in world frame.

    Fidelity terms use OMG own implementations; the standalone physical terms are computed
    on the generated motion alone and so do not depend on the reference at all.
    """
    from omg.benchmarks.metrics.tracking import g_mpjpe, mpjpe, e_vel, e_acc
    from omg.benchmarks.runners.h2h_retarget import (
        compute_foot_sliding, compute_ground_penetration, compute_joint_jump_rate,
        compute_joint_limit_rates,
    )

    dt = 1.0 / float(fps)
    n = min(len(gen_q), len(gt_q))
    gq, tq = gen_q[:n], gt_q[:n]
    gp = kin.forward_body_positions(torch.from_numpy(gq).unsqueeze(0).to(dev))[0].cpu().numpy()
    tp = kin.forward_body_positions(torch.from_numpy(tq).unsqueeze(0).to(dev))[0].cpu().numpy()
    lo = kin.joint_lower_limits.detach().cpu().numpy()
    hi = kin.joint_upper_limits.detach().cpu().numpy()

    out = {"frames": int(n)}
    out.update({"mpjpe_mm": mpjpe(gp, tp), "g_mpjpe_mm": g_mpjpe(gp, tp),
                "e_vel": e_vel(gp, tp), "e_acc": e_acc(gp, tp)})
    out.update(compute_joint_limit_rates(gq[:, 7:], lo, hi))
    out["joint_jump_rate"] = compute_joint_jump_rate(gq[:, 7:])
    out.update(compute_ground_penetration(gp, body_ids=list(foot_ids)))
    out.update(compute_foot_sliding(gp, foot_ids, dt))
    out["body_jerk"] = float(np.abs(np.diff(gp, n=3, axis=0)).mean() / dt ** 3)
    out["fall_rate"] = float(gp[:, 0, 2].min() < 0.35)
    out["xy_disp_m"] = float(np.linalg.norm(gq[-1, :2] - gq[0, :2]))
    out["gt_xy_disp_m"] = float(np.linalg.norm(tq[-1, :2] - tq[0, :2]))
    out["disp_ratio"] = out["xy_disp_m"] / max(out["gt_xy_disp_m"], 1e-6)
    return {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in out.items()}


def validate(conv, rep, kin, rows, cache, token, dev, L, H, log):
    """Prove the pipeline before spending GPU time. Any failure raises SystemExit.

    Written because this project has now shipped four shape-correct / scale-wrong bugs
    (SSOT #14, #15, #21, #24). Every one would have been caught by an assertion like these.
    """
    ok = {}

    # V1 -- codec parity on a REAL clip: the same gate the converter itself used.
    row = rows.iloc[0]
    q, lost = clip_qpos_from_source(row, conv, cache, token)
    if q is None:
        raise SystemExit("V1 FAILED: could not load source for clip "
                         + str(row["clip_id"]) + ": " + str(lost))
    rt = conv.verify_roundtrip(rep, q, L, H, str(dev))
    log("  V1 codec round-trip on clip %d: %s" % (int(row["clip_id"]), rt))
    if max(rt.values()) > 1e-3:
        raise SystemExit("V1 FAILED: round-trip error %s exceeds 1e-3" % rt)
    ok["V1_codec_roundtrip"] = {k: float(v) for k, v in rt.items()}

    # V2 -- the decode that broke the last render. Shards are RAW; rep.decode()
    # denormalises. normalize->decode must equal the codec own split_features inverse.
    feats, ap, aq = conv.encode_windows(rep, q, [0], L, H, str(dev))
    raw = torch.from_numpy(feats).to(dev)
    apt = torch.from_numpy(ap).to(dev)
    aqt = torch.from_numpy(aq).to(dev)
    a_path = rep.compose_qpos_36(rep.decode(rep.normalize_features(raw)), apt, aqt)
    b_path = rep.codec.decode_to_world_qpos36(rep.codec.split_features(raw), apt, aqt)
    d_ok = float((a_path - b_path).abs().max())
    wrong = rep.compose_qpos_36(rep.decode(raw), apt, aqt)
    d_bug = float((wrong - b_path).abs().max())
    log("  V2 decode identity: normalize->decode vs codec inverse  max|d| = %.3e" % d_ok)
    log("     unnormalised decode (the old bug) differs by         %.3e" % d_bug)
    if d_ok > 1e-4:
        raise SystemExit("V2 FAILED: decode paths disagree by %.3e" % d_ok)
    if d_bug < 1e-3:
        raise SystemExit("V2 FAILED: the known-bad path did not differ; test is insensitive")
    ok["V2_decode_identity"] = {"correct_max_abs": d_ok, "buggy_path_max_abs": d_bug}

    # V3 -- ground truth is source mocap and must actually move.
    gp = kin.forward_body_positions(torch.from_numpy(q).unsqueeze(0).to(dev))[0].cpu().numpy()
    b2i = dict(kin.body_name_to_index)
    lf, rf = b2i["left_ankle_roll_link"], b2i["right_ankle_roll_link"]
    swing = float(max(np.ptp(gp[:, lf] - gp[:, 0], axis=0).max(),
                      np.ptp(gp[:, rf] - gp[:, 0], axis=0).max()))
    disp = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
    jrange = float(np.ptp(q[:, 7:], axis=0).max())
    log("  V3 GT motion, clip %d (%d frames): root %.3f m | foot swing vs pelvis %.3f m "
        "| joint range %.2f rad" % (int(row["clip_id"]), len(q), disp, swing, jrange))
    if swing < 0.05 and jrange < 0.15:
        raise SystemExit("V3 FAILED: source GT is nearly frozen; the loader is wrong")
    ok["V3_gt_motion"] = {"root_disp_m": disp, "foot_swing_m": swing,
                          "joint_range_rad": jrange, "frames": int(len(q))}
    return ok


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--csv", default="/workspace/IndicVLA/OMG/splits_amass_g1_13k/test.csv")
    ap_.add_argument("--arms", default="gt,omg_hi,maila_hi,maila_en")
    ap_.add_argument("--n", type=int, default=6)
    ap_.add_argument("--max-seconds", type=float, default=10.0)
    ap_.add_argument("--cfg-scale", type=float, default=2.0)
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--width", type=int, default=520)
    ap_.add_argument("--height", type=int, default=440)
    ap_.add_argument("--models", default="/workspace/models")
    ap_.add_argument("--ckpt-hi", required=True)
    ap_.add_argument("--ckpt-en", required=True)
    ap_.add_argument("--calib-hi", default="/workspace/adapter_calibration.pt")
    ap_.add_argument("--calib-en", default="/workspace/adapter_calibration_en.pt")
    ap_.add_argument("--cache", default="/workspace/cache/g1_npz")
    ap_.add_argument("--out", default="/workspace/full_panels")
    ap_.add_argument("--validate-only", action="store_true")
    a = ap_.parse_args()

    import imageio.v2 as imageio
    from PIL import ImageFont
    import convert_g1_npz_to_omg125 as conv
    from maila_train import build
    from maila_encoder import MurilAdapterEncoder
    from omg.generation.conditions.t5 import FrozenT5TextEncoder

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS_ALL]
    if bad:
        raise SystemExit("unknown arms %s; choose from %s" % (bad, ARMS_ALL))
    L, H = 10, 60
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    token = Path("/root/.creds/motion_hf_token").read_text().strip()
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    df = pd.read_csv(a.csv, low_memory=False)
    df = df[pd.to_numeric(df["duration_s"], errors="coerce") >= 6.0].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    pick = df.iloc[sorted(rng.choice(len(df), size=min(a.n * 4, len(df)),
                                     replace=False))].reset_index(drop=True)

    model = build("100m", a.models + "/omg/checkpoints/updated/100m/sstep=170000.ckpt",
                  a.models + "/t5-base-local", a.models + "/muril-base-cased",
                  dev, True, True, 0.0)
    rep, kin = model.representation, model.representation.kinematics
    base_enc = model.text_encoder
    b2i = dict(kin.body_name_to_index)
    foot_ids = (b2i["left_ankle_roll_link"], b2i["right_ankle_roll_link"])

    log("=== VALIDATION ===")
    checks = validate(conv, rep, kin, pick, cache, token, dev, L, H, log)

    encs = {}
    if any(x.startswith("omg_") for x in arms):
        encs["t5"] = FrozenT5TextEncoder(model_name=a.models + "/t5-base-local",
                                         max_length=50, output_dim=768).to(dev).eval()

    def adapter(calib, ck):
        e = MurilAdapterEncoder(muril_name=a.models + "/muril-base-cased",
                                t5_model_name=a.models + "/t5-base-local",
                                t5_calibration=calib, norm_mode="corpus").to(dev).eval()
        e.adapter.load_state_dict(torch.load(ck, map_location="cpu",
                                             weights_only=False)["adapter"])
        e.adapter.eval()
        return e

    if "maila_hi" in arms:
        encs["maila_hi"] = adapter(a.calib_hi, a.ckpt_hi)
    if "maila_en" in arms:
        encs["maila_en"] = adapter(a.calib_en, a.ckpt_en)

    def make_batch(w, apk, aqk, caption):
        return {
            "history_features": w[:, :L], "prev_state_features": w[:, :L],
            "motion_features": w[:, L:],
            "canon_root_pos": torch.from_numpy(apk).to(dev),
            "canon_root_quat": torch.from_numpy(aqk).to(dev),
            "mask": {"valid": torch.ones(1, H, dtype=torch.bool, device=dev)},
            "has_text": torch.ones(1, dtype=torch.bool, device=dev),
            "fps": torch.tensor([30.0], device=dev),
            "caption": [caption],
        }

    # V4 -- closed-loop rollout really produces the frames asked for, past one chunk.
    row0 = pick.iloc[0]
    q0, _ = clip_qpos_from_source(row0, conv, cache, token)
    n_req = min(int(a.max_seconds * 30), len(q0) - L)
    f0, ap0, aq0 = conv.encode_windows(rep, q0, [0], L, H, str(dev))
    probe_arm = arms[-1] if arms[-1] != "gt" else arms[0]
    model.text_encoder = encs["t5"] if probe_arm.startswith("omg_") else encs[probe_arm]
    with torch.no_grad():
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        g0 = model.generate(make_batch(torch.from_numpy(f0).to(dev), ap0, aq0,
                                       str(row0["caption_1"])),
                            num_frames=n_req, cfg_scale=a.cfg_scale)
    model.text_encoder = base_enc
    got = int(g0["qpos_36"].shape[1])
    chunk = int(rep.sequence_length)
    log("  V4 closed loop: asked %d frames, got %d, chunk=%d -> %d rollout steps"
        % (n_req, got, chunk, int(np.ceil(n_req / chunk))))
    if got != n_req:
        raise SystemExit("V4 FAILED: asked %d frames, generate returned %d" % (n_req, got))
    if n_req <= chunk:
        raise SystemExit("V4 FAILED: %d frames fits one chunk; not a closed-loop test" % n_req)
    checks["V4_closed_loop"] = {"asked": n_req, "got": got, "chunk": chunk}

    # V5 -- label/column ordering cannot drift: both come from one ordered list.
    log("  V5 label order fixed to %s (header and frames share one list)" % arms)
    checks["V5_label_order"] = arms

    (out / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    log("=== VALIDATION PASSED ===")
    if a.validate_only:
        (out / "validation.log").write_text("\n".join(lines), encoding="utf-8")
        return 0

    fonts = {"hi": ImageFont.truetype("/workspace/fonts/dev.ttf", 23),
             "en": ImageFont.truetype("/workspace/fonts/lat.ttf", 23),
             "label": ImageFont.truetype("/workspace/fonts/lat.ttf", 20)}
    from omg.render.mujoco import render_qpos_video

    # Resume: a crash on clip 6 previously destroyed the metrics for clips 1-5 because the
    # manifest was only written after the whole loop. Each clip now persists its own record
    # the moment it completes, and a re-run skips anything already on disk.
    manifest, done, skipped = [], 0, []
    for f in sorted(out.glob("clip_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if (out / rec["file"]).exists():
            manifest.append(rec)
            done += 1
    if manifest:
        log("  resuming: %d clip(s) already complete" % done)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for _, row in pick.iterrows():
            if done >= a.n:
                break
            cid = int(row["clip_id"])
            if any(m["clip_id"] == cid for m in manifest):
                continue
            gt_q, lost = clip_qpos_from_source(row, conv, cache, token)
            if gt_q is None or len(gt_q) < L + H:
                skipped.append({"clip_id": cid, "reason": str(lost)})
                continue
            n_gen = min(int(a.max_seconds * 30), len(gt_q) - L)
            log("    clip %d: source %d frames, generating %d (trunc %.2f)"
                % (cid, len(gt_q), n_gen, lost if isinstance(lost, float) else -1))
            feats, apk, aqk = conv.encode_windows(rep, gt_q, [0], L, H, str(dev))
            w = torch.from_numpy(feats).to(dev)
            caps = {"en": str(row["caption_1"]), "hi": str(row["caption_1_hi"])}
            gt_seg = gt_q[L:L + n_gen]

            qpos, mets = {}, {}
            for arm in arms:
                if arm == "gt":
                    qpos[arm] = gt_seg
                    continue
                model.text_encoder = encs["t5"] if arm.startswith("omg_") else encs[arm]
                try:
                    with torch.no_grad():
                        torch.manual_seed(a.seed)
                        torch.cuda.manual_seed_all(a.seed)
                        g = model.generate(make_batch(w, apk, aqk, caps[LANG_OF[arm]]),
                                           num_frames=n_gen, cfg_scale=a.cfg_scale)
                except Exception as exc:
                    model.text_encoder = base_enc
                    failed = "%s: %s" % (type(exc).__name__, str(exc)[:120])
                    log("    SKIP clip %d (%s failed) %s" % (cid, arm, failed))
                    skipped.append({"clip_id": cid, "arm": arm, "reason": failed})
                    qpos = None
                    break
                model.text_encoder = base_enc
                qpos[arm] = g["qpos_36"][0].float().cpu().numpy()
                mets[arm] = arm_metrics(qpos[arm], gt_seg, kin, foot_ids, 30.0, dev)
            if qpos is None:
                continue

            frames = {}
            for arm in arms:
                p = tmp / ("%d_%s.mp4" % (cid, arm))
                render_qpos_video(qpos[arm], str(p), fps=30, width=a.width, height=a.height,
                                  title="", overlay_lines=None)
                frames[arm] = imageio.mimread(str(p), memtest=False)
            nf = min(len(frames[x]) for x in arms)
            cols = [(arm, "" if arm == "gt" else caps[LANG_OF[arm]]) for arm in arms]
            hdr = draw_header(a.width * len(arms), cols, fonts)
            name = "full_clip%d.mp4" % cid
            with imageio.get_writer(str(out / name), fps=30, quality=8) as wtr:
                for t in range(nf):
                    strip = np.concatenate([frames[arm][t][:, :, :3] for arm in arms], axis=1)
                    wtr.append_data(np.concatenate([hdr, strip], axis=0))
            rec = {"file": name, "clip_id": cid,
                   "source": str(row["source_amass"]), "arms": arms,
                   "seconds": round(nf / 30.0, 2), "frames": int(nf),
                   "truncation_frac": lost,
                   "caption_en": caps["en"], "caption_hi": caps["hi"],
                   "metrics": mets}
            manifest.append(rec)
            (out / ("clip_%d.json" % cid)).write_text(
                json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            log("  [%d/%d] %s  %d frames (%.1fs)  %s"
                % (done + 1, a.n, name, nf, nf / 30.0,
                   "  ".join("%s mpjpe=%.0fmm" % (k, v["mpjpe_mm"]) for k, v in mets.items())))
            done += 1

    agg = {}
    for arm in arms:
        if arm == "gt" or not manifest:
            continue
        keys = [k for k, v in manifest[0]["metrics"][arm].items()
                if isinstance(v, (int, float))]
        agg[arm] = {k: round(float(np.mean([m["metrics"][arm][k] for m in manifest])), 5)
                    for k in keys}
    (out / "manifest.json").write_text(
        json.dumps({"clips": manifest, "aggregate": agg, "validation": checks,
                    "skipped": skipped},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "validation.log").write_text("\n".join(lines), encoding="utf-8")
    log("\nwrote %d full-length panels to %s" % (done, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
