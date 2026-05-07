"""
Pragya-VLA pilot review: Streamlit player for the rendered G1 motion videos.

Run:
    streamlit run player_app.py -- \\
        --videos_dir kimodo_outputs_full/videos_g1 \\
        --manifest  kimodo_outputs_full/manifest.jsonl \\
        --csv       prompts_rewritten.csv

You can also bake those defaults into the file (see DEFAULT_* below) and just
run `streamlit run player_app.py`.

Features:
    * Sidebar list of all 200 prompts (filterable by motion family).
    * Click a prompt to load its video.
    * Prompt header at the top: imperative + Kimodo descriptive form + tags.
    * Native HTML5 video player (play/pause, scrub, fullscreen, speed).
    * Prev / Next buttons.
    * Left/Right arrow keys also navigate (via streamlit-shortcuts).

Dependencies (laptop):
    pip install streamlit pandas
    pip install streamlit-shortcuts        # optional, enables arrow-key nav
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


# ---- Defaults: edit if you don't want to pass CLI flags every time --------
DEFAULT_VIDEOS_DIR = "kimodo_outputs_full/videos_g1"
DEFAULT_MANIFEST = "kimodo_outputs_full/manifest.jsonl"
DEFAULT_CSV = "prompts_rewritten.csv"


# ---- Optional dep: arrow-key shortcuts ------------------------------------
try:
    from streamlit_shortcuts import button as shortcut_button
    HAVE_SHORTCUTS = True
except ImportError:
    HAVE_SHORTCUTS = False


# ---------------------------------------------------------------------------
@dataclass
class Clip:
    prompt_id: str
    imperative_en: str
    kimodo_input: str
    motion_family: str
    primitive_tag: str
    duration_s: float
    video_path: Path


def parse_cli_args() -> argparse.Namespace:
    """Parse args after Streamlit's `--` separator. Falls back to defaults."""
    # Streamlit passes user args after a literal `--` into sys.argv.
    # We tolerate the case where no args were given (running from IDE).
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        user = sys.argv[idx + 1:]
    else:
        user = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default=DEFAULT_VIDEOS_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    return ap.parse_args(user)


@st.cache_data(show_spinner=False)
def load_clips(videos_dir: str, manifest_path: str, csv_path: str) -> list[Clip]:
    """Build the list of clips by joining the manifest with the actual MP4s.

    Falls back to the prompts CSV if the manifest is missing.
    """
    vdir = Path(videos_dir)

    # Prefer manifest (richer fields per prompt)
    rows: dict[str, dict] = {}
    mpath = Path(manifest_path)
    if mpath.exists():
        with open(mpath) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rows[rec["prompt_id"]] = rec
                except json.JSONDecodeError:
                    continue

    # Backfill from prompts_rewritten.csv if either manifest is missing or
    # the manifest is missing fields we want to display.
    cpath = Path(csv_path)
    if cpath.exists():
        df = pd.read_csv(cpath)
        for _, r in df.iterrows():
            pid = r["prompt_id"]
            if pid not in rows:
                rows[pid] = {}
            for k in ("imperative_en", "kimodo_input", "motion_family",
                      "primitive_tag", "duration_s"):
                if k not in rows[pid] or rows[pid].get(k) in (None, ""):
                    rows[pid][k] = r.get(k, "")

    clips: list[Clip] = []
    for pid in sorted(rows.keys()):
        vpath = vdir / f"{pid}.mp4"
        if not vpath.exists():
            continue
        r = rows[pid]
        clips.append(Clip(
            prompt_id=pid,
            imperative_en=str(r.get("imperative_en", "")),
            kimodo_input=str(r.get("kimodo_input", "")),
            motion_family=str(r.get("motion_family", "")),
            primitive_tag=str(r.get("primitive_tag", "")),
            duration_s=float(r.get("duration_s", 0) or 0),
            video_path=vpath,
        ))
    return clips


def init_state(clips: list[Clip]) -> None:
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "family_filter" not in st.session_state:
        st.session_state.family_filter = "all"
    # Clamp current_idx if we re-launched with fewer clips
    st.session_state.current_idx = max(
        0, min(st.session_state.current_idx, len(clips) - 1)
    )


def filtered_clips(clips: list[Clip]) -> list[Clip]:
    f = st.session_state.family_filter
    if f == "all":
        return clips
    return [c for c in clips if c.motion_family == f]


def current_clip(clips: list[Clip]) -> Optional[Clip]:
    visible = filtered_clips(clips)
    if not visible:
        return None
    # Map current_idx (over full list) to the closest position in `visible`.
    target_pid = clips[st.session_state.current_idx].prompt_id
    for c in visible:
        if c.prompt_id == target_pid:
            return c
    # Selected clip is filtered out — snap to first visible
    st.session_state.current_idx = clips.index(visible[0])
    return visible[0]


def goto_offset(clips: list[Clip], delta: int) -> None:
    visible = filtered_clips(clips)
    if not visible:
        return
    cur = current_clip(clips)
    pos = visible.index(cur)
    new_pos = (pos + delta) % len(visible)
    st.session_state.current_idx = clips.index(visible[new_pos])


def goto_id(clips: list[Clip], pid: str) -> None:
    for i, c in enumerate(clips):
        if c.prompt_id == pid:
            st.session_state.current_idx = i
            return


# ---------------------------------------------------------------------------
def main():
    args = parse_cli_args()
    st.set_page_config(
        page_title="Pragya-VLA · Pilot Review",
        layout="wide",
        page_icon="🐯",
    )

    clips = load_clips(args.videos_dir, args.manifest, args.csv)
    if not clips:
        st.error(
            f"No videos found. Looked in: `{args.videos_dir}`\n\n"
            "Run `render_videos.py` first to produce the MP4s."
        )
        st.stop()

    init_state(clips)

    # ---- Sidebar: filter + clip list -------------------------------------
    families = ["all"] + sorted({c.motion_family for c in clips if c.motion_family})
    with st.sidebar:
        st.markdown("### Filter")
        st.session_state.family_filter = st.selectbox(
            "Motion family",
            families,
            index=families.index(st.session_state.family_filter),
            label_visibility="collapsed",
        )
        st.divider()

        visible = filtered_clips(clips)
        st.markdown(f"### Clips ({len(visible)} / {len(clips)})")
        cur = current_clip(clips)

        # Compact list of buttons. Highlight the current selection.
        for c in visible:
            is_current = (cur is not None and c.prompt_id == cur.prompt_id)
            label = f"{'▶ ' if is_current else ''}{c.prompt_id} · {c.imperative_en}"
            if st.button(
                label,
                key=f"sel_{c.prompt_id}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                goto_id(clips, c.prompt_id)
                st.rerun()

    # ---- Main panel ------------------------------------------------------
    cur = current_clip(clips)
    if cur is None:
        st.warning("No clips match the current filter.")
        st.stop()

    visible = filtered_clips(clips)
    pos = visible.index(cur)

    # Header with prompt info
    st.markdown(f"## {cur.imperative_en}")
    st.caption(
        f"`{cur.prompt_id}` · "
        f"family: **{cur.motion_family}** · "
        f"primitive: `{cur.primitive_tag}` · "
        f"duration: {cur.duration_s:.0f}s · "
        f"clip {pos + 1} / {len(visible)}"
    )
    with st.expander("Kimodo input (descriptive form)", expanded=False):
        st.code(cur.kimodo_input, language=None)

    # Video
    st.video(str(cur.video_path), autoplay=True, loop=True)

    # ---- Navigation ------------------------------------------------------
    col_prev, col_next, _ = st.columns([1, 1, 6])

    def on_prev():
        goto_offset(clips, -1)

    def on_next():
        goto_offset(clips, +1)

    if HAVE_SHORTCUTS:
        with col_prev:
            shortcut_button(
                "⏮  Previous",
                shortcut="ArrowLeft",
                on_click=on_prev,
                use_container_width=True,
            )
        with col_next:
            shortcut_button(
                "Next  ⏭",
                shortcut="ArrowRight",
                on_click=on_next,
                use_container_width=True,
            )
        st.caption("Tip: ← / → arrow keys also navigate.")
    else:
        with col_prev:
            if st.button("⏮  Previous", use_container_width=True):
                on_prev()
                st.rerun()
        with col_next:
            if st.button("Next  ⏭", use_container_width=True):
                on_next()
                st.rerun()
        st.caption(
            "Install `streamlit-shortcuts` to enable arrow-key navigation: "
            "`pip install streamlit-shortcuts`"
        )


if __name__ == "__main__":
    main()