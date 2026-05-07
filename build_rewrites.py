"""
Pragya-VLA pilot: T1 atomic prompt rewriter.

Converts robot-imperative English ("Stop.", "Raise your left arm.") into
Kimodo-friendly descriptive form ("A person stops.", "A person raises their
left arm.") that matches Kimodo's training distribution.

Strategy: rule-based (deterministic, auditable). Built around a closed
vocabulary of action heads (~50) and a small set of modifiers, which is
sufficient for T1 atomics. For T3+ composites we will need a stronger pass.
"""
import openpyxl
import csv
import re
from pathlib import Path

# ---------- Step 1: strip temporal/discourse hedges (drop entirely) ----------
# These add no motion semantics and confuse the descriptive model.
HEDGES_TO_DROP = [
    # leading
    r"^at once,?\s*",
    r"^immediately,?\s*",
    r"^just\s+",
    # trailing
    r"\s+(?:promptly|directly|right away|immediately|straight away|"
    r"without delay|at once|now|for now|here|as instructed|"
    r"before anything else|in a neutral manner)\.?\s*$",
]

# ---------- Step 2: stance/positional modifiers → canonical "in place" -------
STATIONARY_PATTERNS = [
    r"\s+(?:in place|on the spot|where you stand|where you are|"
    r"without moving away|in your current position)\b",
]

# ---------- Step 3: speed/care modifiers → "carefully" or "slowly" -----------
CAREFUL_PATTERNS = [
    (r"\s+with care\b", " carefully"),
    (r"\s+with control\b", " carefully"),
    (r"\s+in a steady way\b", " steadily"),
]

# ---------- Step 4: action head conjugation (imperative → 3rd person sg) ----
# This is the heart of the rewriter. Each entry handles:
#   imperative form         -> conjugated descriptive form
# Order matters: longer/more-specific patterns first.
#
# Coverage was checked against all 200 unique action stems in the file.
ACTION_REWRITES = [
    # ----- arm/hand gestures (most have "your X arm/hand/hands") -----
    (r"^raise your left arm\b",         "raises their left arm"),
    (r"^raise your right arm\b",        "raises their right arm"),
    (r"^lower your left arm\b",         "lowers their left arm"),
    (r"^lower your right arm\b",        "lowers their right arm"),
    (r"^extend your left arm\b",        "extends their left arm"),
    (r"^extend your right arm\b",       "extends their right arm"),
    (r"^lift your left hand\b",         "lifts their left hand"),
    (r"^lift your right hand\b",        "lifts their right hand"),
    (r"^raise arms\b",                  "raises both arms"),
    (r"^lower arms\b",                  "lowers both arms"),
    (r"^extend arms\b",                 "extends both arms"),
    (r"^cross your arms\b",             "crosses their arms"),
    (r"^open your hands\b",             "opens their hands"),
    (r"^close your hands\b",            "closes their hands"),
    (r"^lift your chin\b",              "lifts their chin"),

    # ----- body posture -----
    (r"^raise your body\b",             "stands up tall"),
    (r"^straighten up\b",               "straightens up to a standing pose"),
    (r"^stand up\b",                    "stands up from a seated pose"),
    (r"^sit down\b",                    "sits down on the ground"),
    (r"^crouch\b",                      "crouches down low"),
    (r"^squat\b",                       "squats down"),
    (r"^duck\b",                        "ducks down low"),  # avoid bird ambiguity
    (r"^bend down\b",                   "bends down"),
    (r"^bow\b",                         "bows forward"),
    (r"^lean forward\b",                "leans forward"),
    (r"^lean back\b",                   "leans backward"),
    (r"^rise\b",                        "stands up from a crouched pose"),
    (r"^square your shoulders\b",       "squares their shoulders"),
    (r"^center your balance\b",         "centers their balance"),
    (r"^plant your feet\b",             "plants their feet firmly"),
    (r"^shift your weight\b",           "shifts their weight from one leg to the other"),
    (r"^steady yourself\b",             "stands stably"),
    (r"^close stance\b",                "stands with feet together"),
    (r"^spread legs\b",                 "stands with their legs spread apart"),

    # ----- head-only motions -----
    (r"^look up\b",                     "looks up"),
    (r"^look down\b",                   "looks down"),
    (r"^look left\b",                   "looks to the left"),
    (r"^look right\b",                  "looks to the right"),

    # ----- facing direction -----
    (r"^face forward\b",                "turns to face forward"),
    (r"^face backward\b",               "turns to face backward"),
    (r"^face left\b",                   "turns to face left"),
    (r"^face right\b",                  "turns to face right"),

    # ----- turning / spinning / pivoting -----
    (r"^turn around\b",                 "turns around 180 degrees"),
    (r"^turn back\b",                   "turns around 180 degrees"),
    (r"^turn left\b",                   "turns to the left"),
    (r"^turn right\b",                  "turns to the right"),
    (r"^rotate around\b",               "rotates around 180 degrees"),
    (r"^rotate back\b",                 "rotates back to face the opposite direction"),
    (r"^rotate left\b",                 "rotates to the left"),
    (r"^rotate right\b",                "rotates to the right"),
    (r"^spin around\b",                 "spins around 360 degrees"),
    (r"^spin back\b",                   "spins around 180 degrees"),
    (r"^spin left\b",                   "spins to the left"),
    (r"^spin right\b",                  "spins to the right"),
    (r"^pivot around\b",                "pivots around 180 degrees"),
    (r"^pivot back\b",                  "pivots back to face the opposite direction"),
    (r"^pivot left\b",                  "pivots to the left"),
    (r"^pivot right\b",                 "pivots to the right"),

    # ----- stepping / locomotion -----
    (r"^step forward\b",                "takes a step forward"),
    (r"^step back\b",                   "takes a step backward"),
    (r"^step left\b",                   "takes a step to the left"),
    (r"^step right\b",                  "takes a step to the right"),
    (r"^step aside\b",                  "steps aside"),
    (r"^step\b",                        "takes a step forward"),
    (r"^back up\b",                     "steps backward"),
    (r"^sidestep left\b",               "sidesteps to the left"),
    (r"^sidestep right\b",              "sidesteps to the right"),
    (r"^move\b",                        "walks forward"),  # context default
    (r"^go\b",                          "walks forward"),

    # ----- stop / hold / pause / freeze -----
    (r"^stop\b",                        "stops walking and stands still"),
    (r"^halt\b",                        "stops walking and stands still"),
    (r"^pause\b",                       "pauses and stands still"),
    (r"^freeze\b",                      "freezes in place"),
    (r"^hold still\b",                  "holds still"),
    (r"^hold position\b",               "holds their position"),
    (r"^stand still\b",                 "stands still"),

    # ----- social gestures -----
    (r"^clap\b",                        "claps their hands"),
    (r"^salute\b",                      "salutes"),
    (r"^nod\b",                         "nods their head"),
    (r"^shrug\b",                       "shrugs their shoulders"),
    (r"^give a thumbs up\b",            "gives a thumbs up"),
    (r"^point forward\b",               "points forward"),

    # ----- misc -----
    (r"^relax\b",                       "relaxes into a neutral pose"),
    (r"^reset\b",                       "stands in a neutral pose"),
]

# Trailing "and X" continuations — applied AFTER main verb conjugation
TRAILING_CONTINUATIONS = [
    (r"\s+and\s+stay\s+ready\b",        " and stays in a ready stance"),
    (r"\s+and\s+hold\b",                " and holds the pose"),
]


def normalize(text: str) -> str:
    """Lowercase, strip trailing period, collapse whitespace."""
    t = text.strip().rstrip(".").lower()
    t = re.sub(r"\s+", " ", t)
    return t


def rewrite(prompt: str) -> str:
    """Main rewrite: imperative robot-natural English → 'A person ...' form."""
    s = normalize(prompt)

    # 1. Drop hedges
    for pat in HEDGES_TO_DROP:
        s = re.sub(pat, " ", s).strip()
    # 2. Drop stationary modifier (we'll add "in place" later if removed)
    had_stationary = False
    for pat in STATIONARY_PATTERNS:
        if re.search(pat, s):
            had_stationary = True
            s = re.sub(pat, "", s).strip()
    # 3. Care/control modifier
    care_suffix = ""
    for pat, repl in CAREFUL_PATTERNS:
        if re.search(pat, s):
            care_suffix = repl
            s = re.sub(pat, "", s).strip()
    # 4. Trailing continuations — capture and re-apply later
    trailing = ""
    for pat, repl in TRAILING_CONTINUATIONS:
        if re.search(pat, s):
            trailing = repl
            s = re.sub(pat, "", s).strip()

    # 5. Action head conjugation
    conjugated = None
    for pat, repl in ACTION_REWRITES:
        if re.match(pat, s):
            conjugated = re.sub(pat, repl, s, count=1)
            break

    if conjugated is None:
        # Fallback: prefix as-is — flag for manual review
        return f"A person {s}.  # FALLBACK_UNMATCHED"

    # 6. Reassemble
    parts = ["A person", conjugated]
    if had_stationary:
        parts.append("in place")
    if care_suffix:
        parts.append(care_suffix.strip())
    if trailing:
        parts.append(trailing.strip())
    out = " ".join(parts).rstrip() + "."
    out = re.sub(r"\s+", " ", out)
    return out


# ---------- Run over all 200 prompts and emit a verification CSV ----------
def main():
    src = Path("/home/claude/translations_200.xlsx")
    out_csv = Path("/home/claude/prompts_rewritten.csv")

    wb = openpyxl.load_workbook(src)
    ws = wb["translations"]

    rows = []
    fallbacks = []
    for r in range(2, ws.max_row + 1):
        pid = ws.cell(r, 1).value
        family = ws.cell(r, 4).value
        en_robot = ws.cell(r, 7).value
        primitive_tag = ws.cell(r, 16).value
        duration = ws.cell(r, 24).value
        kimodo_input = rewrite(en_robot)
        rows.append({
            "prompt_id": pid,
            "motion_family": family,
            "primitive_tag": primitive_tag,
            "duration_s": duration,
            "imperative_en": en_robot,
            "kimodo_input": kimodo_input,
        })
        if "FALLBACK_UNMATCHED" in kimodo_input:
            fallbacks.append((pid, en_robot, kimodo_input))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Fallbacks (unmatched, need manual review): {len(fallbacks)}")
    for pid, orig, fb in fallbacks:
        print(f"  {pid}: '{orig}' -> {fb}")


if __name__ == "__main__":
    main()
