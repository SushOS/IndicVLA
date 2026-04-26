import itertools
import json
import math
import random
import re
from pathlib import Path

random.seed(42)

# Closed vocabulary banks grounded in the paper's prompt grammar.
LOCO_VERBS = ["walk", "move", "go", "step", "proceed", "advance", "march"]
TURN_VERBS = ["turn", "rotate", "pivot", "spin"]
POSTURE_VERBS = [
    "crouch", "kneel", "stand up", "rise", "lean forward", "lean back",
    "bow", "sit down", "straighten up", "lower your body", "raise your body",
    "squat", "duck", "bend down",
]
GESTURE_VERBS = [
    "wave", "point forward", "raise your right arm", "raise your left arm",
    "lower your right arm", "lower your left arm", "clap", "nod",
    "shrug", "cross your arms", "extend your right arm", "extend your left arm",
    "lift your right hand", "lift your left hand", "salute", "give a thumbs up",
]
STOP_VERBS = ["stop", "halt", "freeze", "pause", "hold position", "stand still"]

DIRECTIONS = [
    "forward", "backward", "left", "right",
    "to the left", "to the right", "ahead", "back",
    "diagonally left", "diagonally right",
]
TURN_DIRS = ["left", "right", "around", "back"]
TURN_EXTENTS = ["slightly", "sharply", "a little", "halfway", "completely"]
SPEEDS = ["slowly", "carefully", "steadily", "quickly", "cautiously"]
COUNTS = [
    "one step", "two steps", "three steps", "four steps", "five steps",
    "half a metre", "one metre", "two metres", "three metres",
]
BODY_SIDES = [
    "your left arm", "your right arm",
    "your left hand", "your right hand",
    "your left leg", "your right leg",
]

LANDMARKS = [
    "the door", "the gate", "the marker", "the wall",
    "the corridor", "the table", "the chair", "the ramp",
    "the cone", "the barrier", "the pillar", "the window",
    "the shelf", "the entrance", "the exit", "the desk",
    "the cabinet", "the sofa", "the bench", "the post",
]
PATH_TYPES = [
    "the straight path", "the curved path", "the zigzag path",
    "the marked route", "the narrow passage", "the designated path",
    "the indicated route", "the clear path",
]
WAYPOINTS = [
    "waypoint A", "waypoint B", "waypoint C", "waypoint D",
    "the first marker", "the second marker", "the third marker",
    "the checkpoint", "the target point", "the goal position",
]
OBSTACLES = [
    "the low bar", "the gap", "the puddle", "the step",
    "the obstacle", "the barrier", "the debris", "the box",
    "the pole", "the cable", "the raised platform",
]
OBS_VERBS = [
    "duck under", "step over", "go around", "move past",
    "climb over", "squeeze through", "navigate around",
]

SAFETY_CONDS = [
    "if the path is blocked", "if it is unsafe",
    "if you detect an obstacle", "when the path is clear",
    "before crossing", "if uncertain", "if the floor is wet",
    "if there is a person nearby", "if the surface is uneven",
    "when visibility is low", "if the passage is narrow",
    "before proceeding", "if the way is obstructed",
]
SAFETY_ACTS = [
    "stop and wait", "stop immediately", "slow down",
    "inspect first", "retreat one step", "proceed carefully",
    "wait for clearance", "halt and assess", "pause and look around",
    "back up and reassess", "reduce speed", "stand still",
]
SOCIAL_CONDS = [
    "near a person", "when a person approaches",
    "when someone is in the way", "if a person is nearby",
    "when passing a person",
]
SOCIAL_ACTS = [
    "yield", "slow down", "keep your distance",
    "pause and wait", "move to the side", "stop briefly",
    "step aside", "give way",
]
TERRAIN_SURFS = [
    "the wet floor", "the uneven surface", "the loose gravel",
    "the slippery tiles", "the rough ground", "the narrow ramp",
    "the steep incline", "the unstable surface",
]

TIER_SPECS = {
    "T1": {
        "target": 2400,
        "candidate_quota": 2500,
        "review_rate": 0.12,
        "qc_focus": "lexical clarity, semantic uniqueness",
        "duration_s": 3.0,
    },
    "T2": {
        "target": 1600,
        "candidate_quota": 1700,
        "review_rate": 0.12,
        "qc_focus": "parameter preservation, modifier consistency",
        "duration_s": 3.5,
    },
    "T3": {
        "target": 2000,
        "candidate_quota": 2150,
        "review_rate": 0.22,
        "qc_focus": "action order, temporal coherence",
        "duration_s": 5.0,
    },
    "T4": {
        "target": 1400,
        "candidate_quota": 1500,
        "review_rate": 0.22,
        "qc_focus": "constraint compliance, path fidelity",
        "duration_s": 6.5,
    },
    "T5": {
        "target": 1200,
        "candidate_quota": 1350,
        "review_rate": 0.35,
        "qc_focus": "landmark grounding, reference clarity",
        "duration_s": 5.5,
    },
    "T6": {
        "target": 800,
        "candidate_quota": 900,
        "review_rate": 0.35,
        "qc_focus": "safety semantics, abstention fidelity",
        "duration_s": 6.5,
    },
    "T7": {
        "target": 600,
        "candidate_quota": 750,
        "review_rate": 0.35,
        "qc_focus": "long-range consistency, subgoal completion",
        "duration_s": 9.0,
    },
}


def sentence_case(text: str) -> str:
    return text[0].upper() + text[1:]


def dedupe_casefold(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        cleaned = " ".join(item.strip().split())
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def strip_terminal_period(text: str) -> str:
    return text[:-1] if text.endswith(".") else text


def canonical_to_robot_natural(prompt: str) -> str:
    text = strip_terminal_period(prompt).strip()

    # Drop neutral adverbial wrappers from canonical forms.
    leading_patterns = [
        r"^Immediately\s+",
        r"^At once,\s+",
        r"^Just\s+",
    ]
    trailing_patterns = [
        r"\s+for now$",
        r"\s+straight away$",
        r"\s+promptly$",
        r"\s+directly$",
        r"\s+now$",
        r"\s+immediately$",
        r"\s+right away$",
        r"\s+at once$",
        r"\s+in place$",
        r"\s+here$",
        r"\s+without delay$",
        r"\s+where you are$",
        r"\s+precisely$",
        r"\s+with control$",
    ]
    for pattern in leading_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    for pattern in trailing_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    replacements = [
        (r"^Walk\b", "Move"),
        (r"^Go\b", "Move"),
        (r"^Proceed\b", "Move"),
        (r"^Advance\b", "Move"),
        (r"^March\b", "Move"),
        (r"\bgo to\b", "move to"),
        (r"\bgo toward\b", "move toward"),
        (r"\bcontinue to\b", "move to"),
        (r"\bturn back\b", "turn around"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return sentence_case(text) + "."


def add_templates(out: set[str], phrases: list[str], templates: list[str]) -> None:
    for phrase in phrases:
        sentence = sentence_case(phrase)
        for template in templates:
            out.add(template.format(phrase=phrase, sentence=sentence))


def build_translation_stub() -> dict:
    return {
        "hi": {"literal": None, "robot_natural": None},
        "bn": {"literal": None, "robot_natural": None},
        "te": {"literal": None, "robot_natural": None},
    }


def sample_review_indices(count: int, rate: float) -> set[int]:
    review_count = min(count, math.ceil(count * rate))
    return set(range(review_count))


def constrain_pool(unique: list[str], quota: int) -> list[str]:
    shuffled = unique[:]
    random.shuffle(shuffled)
    return shuffled[:quota]


def gen_t1() -> list[str]:
    atomic_phrases = set()
    atomic_phrases.update(STOP_VERBS)
    atomic_phrases.update(POSTURE_VERBS)
    atomic_phrases.update(GESTURE_VERBS)
    atomic_phrases.update(LOCO_VERBS)

    for tv in TURN_VERBS:
        for d in TURN_DIRS:
            atomic_phrases.add(f"{tv} {d}")

    atomic_phrases.update([
        "step forward", "step back", "step aside", "step left", "step right",
        "sidestep left", "sidestep right", "back up",
        "face left", "face right", "face forward", "face backward",
        "look left", "look right", "look up", "look down", "look around",
        "raise arms", "lower arms", "extend arms", "spread legs", "close stance",
        "open your hands", "close your hands", "shift your weight",
        "center your balance", "square your shoulders", "plant your feet",
        "lift your chin", "drop your shoulders", "steady yourself",
        "hold still", "breathe", "brace", "relax", "reset",
    ])

    templates = [
        "{sentence}.",
        "{sentence} now.",
        "Immediately {phrase}.",
        "{sentence} immediately.",
        "{sentence} right away.",
        "{sentence} at once.",
        "{sentence} in place.",
        "{sentence} here.",
        "{sentence} without delay.",
        "At once, {phrase}.",
        "{sentence} where you are.",
        "{sentence} directly.",
        "{sentence} promptly.",
        "{sentence} straight away.",
        "Just {phrase}.",
        "{sentence} for now.",
        "{sentence} where you stand.",
        "{sentence} in your current position.",
        "{sentence} without moving away.",
        "{sentence} with control.",
        "{sentence} as instructed.",
        "{sentence} on the spot.",
        "{sentence} before anything else.",
        "{sentence} and hold.",
        "{sentence} with care.",
        "{sentence} in a steady way.",
        "{sentence} and stay ready.",
        "{sentence} in a neutral manner.",
    ]

    out = set()
    add_templates(out, sorted(atomic_phrases), templates)
    return list(out)


def gen_t2() -> list[str]:
    parameterized = set()

    for lv in LOCO_VERBS:
        for d in DIRECTIONS:
            parameterized.add(f"{lv} {d}")

    for lv in ["walk", "move", "step", "go"]:
        for c in COUNTS:
            parameterized.add(f"{lv} {c}")

    for lv in ["walk", "move", "step"]:
        for d in ["forward", "backward", "left", "right"]:
            for c in COUNTS:
                parameterized.add(f"{lv} {d} {c}")

    for lv in ["walk", "move", "proceed", "advance"]:
        for sp in SPEEDS:
            parameterized.add(f"{lv} {sp}")

    for lv in ["walk", "move", "step"]:
        for d in ["forward", "backward", "left", "right"]:
            for sp in SPEEDS:
                parameterized.add(f"{lv} {d} {sp}")

    for tv in TURN_VERBS:
        for d in ["left", "right"]:
            parameterized.add(f"{tv} {d}")
            for sp in ["slowly", "quickly", "carefully", "steadily"]:
                parameterized.add(f"{tv} {d} {sp}")
            for ext in TURN_EXTENTS:
                parameterized.add(f"{tv} {d} {ext}")

    for action in ["raise", "lower", "extend", "lift", "swing"]:
        for side in BODY_SIDES:
            parameterized.add(f"{action} {side}")

    for pv in ["lean", "tilt", "bend"]:
        for d in ["left", "right", "forward", "backward"]:
            parameterized.add(f"{pv} {d}")

    for d in ["left", "right"]:
        for c in ["one step", "two steps", "three steps"]:
            parameterized.add(f"sidestep {d} {c}")
        for sp in ["slowly", "quickly", "carefully"]:
            parameterized.add(f"sidestep {d} {sp}")

    templates = [
        "{sentence}.",
        "{sentence} precisely.",
        "{sentence} with control.",
        "{sentence} now.",
        "{sentence} carefully.",
        "{sentence} steadily.",
    ]

    out = set()
    add_templates(out, sorted(parameterized), templates)
    return list(out)


def gen_t3() -> list[str]:
    out = set()
    two_step = set()
    three_step = set()

    for lv in ["walk", "move", "step"]:
        for d in ["forward", "ahead"]:
            for tv in ["turn left", "turn right", "turn around"]:
                two_step.add((f"{lv} {d}", tv))

    for td in ["turn left", "turn right"]:
        for lv in ["walk forward", "move ahead", "proceed forward"]:
            two_step.add((td, lv))

    for lv in LOCO_VERBS:
        for d in ["forward", "ahead", "backward"]:
            two_step.add((f"{lv} {d}", "stop"))

    for pv in ["crouch", "kneel", "squat", "duck", "bow"]:
        for lv in ["move forward", "walk forward", "step ahead"]:
            two_step.add((pv, lv))
        two_step.add((pv, "stand up"))
        two_step.add((pv, "rise"))

    for gv in ["wave", "clap", "nod", "salute", "point forward"]:
        for lv in ["walk forward", "move forward", "step ahead"]:
            two_step.add((gv, lv))
            two_step.add((lv, gv))
        two_step.add((gv, "stop"))

    for lv in ["walk", "move", "step"]:
        for c in ["two steps", "three steps", "one metre", "two metres"]:
            for td in ["turn left", "turn right"]:
                two_step.add((f"{lv} forward {c}", td))
                two_step.add((f"{lv} forward {c}", "stop"))

    for d in ["left", "right"]:
        for lv in ["walk forward", "move forward", "step ahead"]:
            two_step.add((f"sidestep {d}", lv))
        for td in ["turn left", "turn right"]:
            two_step.add((f"sidestep {d}", td))

    three_step_fixed = [
        "Walk forward, turn left, and stop.",
        "Walk forward, turn right, and stop.",
        "Crouch, move ahead, then stand up.",
        "Turn right, wave, and stop.",
        "Sidestep left, move forward, then stop.",
        "Back up, turn around, and walk forward.",
        "Wave, step forward, then stop.",
        "Crouch, move forward three steps, then rise.",
        "Turn left, walk forward, then turn right.",
        "Stop, look around, then move forward.",
        "Turn right, sidestep left, then stop.",
        "Kneel, stand up, then walk forward.",
        "Move backward, turn around, then walk forward.",
        "Walk forward two steps, turn right, and stop.",
        "Walk forward three steps, turn left, and stop.",
        "Turn left, walk forward, and stop.",
        "Turn right, walk forward, and stop.",
        "Move forward, clap, then stop.",
        "Crouch, walk forward, then stand up.",
        "Move forward, nod, then stop.",
    ]
    out.update(three_step_fixed)

    for lv in ["walk", "move", "step"]:
        for c in ["two steps", "three steps", "one metre"]:
            for td in ["turn left", "turn right"]:
                three_step.add((f"{lv} forward {c}", td, "stop"))
                three_step.add((f"{lv} forward {c}", td, "walk forward"))

    for td1 in ["turn left", "turn right"]:
        for lv in ["walk forward", "move ahead", "step forward"]:
            for td2 in ["turn left", "turn right", "turn around"]:
                three_step.add((td1, lv, td2))

    for pv in ["crouch", "kneel", "squat", "duck"]:
        for lv in ["move forward", "walk ahead", "step forward two steps"]:
            for fin in ["stand up", "rise", "stop"]:
                three_step.add((pv, lv, fin))

    for lv in ["walk forward", "move ahead", "step forward", "advance forward"]:
        for gv in ["wave", "clap", "nod", "salute"]:
            three_step.add((lv, gv, "stop"))

    for gv1 in ["wave", "clap", "nod", "point forward"]:
        for lv in ["walk forward", "move ahead", "step forward"]:
            for gv2 in ["stop", "salute", "wave", "nod"]:
                three_step.add((gv1, lv, gv2))

    for d in ["left", "right"]:
        for lv in ["walk forward", "move ahead", "step forward"]:
            for fin in ["stop", "turn left", "turn right"]:
                three_step.add((f"sidestep {d}", lv, fin))

    for a, b in sorted(two_step):
        out.add(f"{sentence_case(a)} and {b}.")
        out.add(f"{sentence_case(a)}, then {b}.")
        out.add(f"{sentence_case(a)}; then {b}.")
        out.add(f"{sentence_case(a)}, and after that {b}.")
        out.add(f"{sentence_case(a)}, followed by {b}.")
        out.add(f"{sentence_case(a)}. Then {sentence_case(b)}.")
        out.add(f"First {a}, then {b}.")

    for a, b, c in sorted(three_step):
        out.add(f"{sentence_case(a)}, {b}, and {c}.")
        out.add(f"{sentence_case(a)}, then {b}, then {c}.")
        out.add(f"{sentence_case(a)}; then {b}; then {c}.")
        out.add(f"{sentence_case(a)}, after that {b}, then {c}.")
        out.add(f"First {a}, then {b}, then {c}.")
        out.add(f"{sentence_case(a)}, followed by {b}, and finally {c}.")
        out.add(f"{sentence_case(a)}. Then {sentence_case(b)}. Then {sentence_case(c)}.")

    return list(out)


def gen_t4() -> list[str]:
    out = set()

    for pt in PATH_TYPES:
        out.add(f"Follow {pt}.")
        out.add(f"Follow {pt} and stop.")
        out.add(f"Follow {pt} to the end.")
        out.add(f"Follow {pt} to the end and stop.")
        out.add(f"Stay on {pt}.")
        out.add(f"Stay on {pt} until the end.")
        for lm in LANDMARKS:
            out.add(f"Follow {pt} and stop at {lm}.")
            out.add(f"Follow {pt} until you reach {lm}, then stop.")
            out.add(f"Stay on {pt} and stop near {lm}.")
            out.add(f"Use {pt} to reach {lm}, then hold position.")

    for i, wp_a in enumerate(WAYPOINTS):
        for j, wp_b in enumerate(WAYPOINTS):
            if i >= j:
                continue
            out.add(f"Move to {wp_a}, then to {wp_b}.")
            out.add(f"Move to {wp_a}, then to {wp_b}, then stop.")
            out.add(f"Navigate to {wp_a}, then proceed to {wp_b}.")
            out.add(f"Reach {wp_a}, continue to {wp_b}, and hold.")
            out.add(f"Go by way of {wp_a}, then finish at {wp_b}.")

    for wp_a, wp_b, wp_c in itertools.permutations(WAYPOINTS[:6], 3):
        out.add(f"Move to {wp_a}, then to {wp_b}, then to {wp_c}.")
        out.add(f"Go to {wp_a}, then {wp_b}, then {wp_c}, and stop.")
        out.add(f"Navigate through {wp_a}, then {wp_b}, and finish at {wp_c}.")

    for ov in OBS_VERBS:
        for ob in OBSTACLES:
            out.add(f"{sentence_case(ov)} {ob}.")
            out.add(f"{sentence_case(ov)} {ob} and continue.")
            out.add(f"{sentence_case(ov)} {ob}, then move forward.")
            out.add(f"{sentence_case(ov)} {ob} and stop after clearing it.")
        for pt in PATH_TYPES:
            out.add(f"{sentence_case(ov)} the obstacle along {pt}.")
            out.add(f"{sentence_case(ov)} the obstacle, return to {pt}, and continue.")

    end_pose = [
        "Reach the end of the path and stop in a neutral stance.",
        "Stop at the marked position and face forward.",
        "Walk to the target position and crouch.",
        "Move forward until the waypoint and hold position.",
        "Follow the path and end at the goal pose.",
        "Proceed to the goal marker and freeze.",
        "Walk to the keyframe position and stop.",
        "Move to the target position and face the marker.",
        "Reach the goal and lower your body into a crouch.",
        "Move to the endpoint and stand upright.",
        "Move to the final marker and bow.",
        "Go to the indicated position and kneel.",
    ]
    out.update(end_pose)

    for pt in ["the dense path", "the marked route", "the indicated path"]:
        for wp in WAYPOINTS:
            out.add(f"Follow {pt} to {wp} and stop.")
            out.add(f"Follow {pt}, pass {wp}, and continue.")
            out.add(f"Stay on {pt}, reach {wp}, and hold position.")
        for lm in LANDMARKS:
            out.add(f"Follow {pt} and stop near {lm}.")
            out.add(f"Use {pt} to approach {lm}, then stop.")

    for pt in PATH_TYPES:
        for wp in WAYPOINTS:
            for lm in LANDMARKS[:8]:
                out.add(f"Follow {pt} to {wp}, continue to {lm}, and stop.")
                out.add(f"Stay on {pt}, pass {wp}, and finish near {lm}.")

    return list(out)


def gen_t5() -> list[str]:
    out = set()

    for lm in LANDMARKS:
        for lv in ["move", "walk", "go", "proceed", "advance"]:
            out.add(f"{sentence_case(lv)} to {lm}.")
            out.add(f"{sentence_case(lv)} to {lm} and stop.")
            out.add(f"{sentence_case(lv)} toward {lm}.")
            out.add(f"{sentence_case(lv)} toward {lm} and wait.")

        for d in ["left", "right"]:
            out.add(f"Turn {d} at {lm}.")
            out.add(f"When you reach {lm}, turn {d}.")
            out.add(f"At {lm}, turn {d}.")

        for lv in ["pass", "walk past", "move past", "go past"]:
            out.add(f"{sentence_case(lv)} {lm}.")
            out.add(f"{sentence_case(lv)} {lm} and continue forward.")
            out.add(f"{sentence_case(lv)} {lm} and stop.")

        for sv in ["stop", "halt", "pause"]:
            out.add(f"{sentence_case(sv)} near {lm}.")
            out.add(f"{sentence_case(sv)} at {lm}.")
            out.add(f"{sentence_case(sv)} beside {lm}.")
            out.add(f"{sentence_case(sv)} in front of {lm}.")

    for lm_a, lm_b in itertools.combinations(LANDMARKS, 2):
        out.add(f"Pass {lm_a} and stop near {lm_b}.")
        out.add(f"Turn left at {lm_a} and walk to {lm_b}.")
        out.add(f"Turn right at {lm_a} and walk to {lm_b}.")
        out.add(f"Move from {lm_a} to {lm_b} and stop.")

    return list(out)


def gen_t6() -> list[str]:
    out = set()

    for cond in SAFETY_CONDS:
        for act in SAFETY_ACTS:
            out.add(f"{sentence_case(cond)}, {act}.")
            out.add(f"{sentence_case(cond)}, {act} before continuing.")

    for lv in ["move", "walk", "proceed", "advance"]:
        for sp in ["slowly", "carefully", "cautiously", "steadily"]:
            for ts in TERRAIN_SURFS:
                out.add(f"{sentence_case(lv)} {sp} over {ts}.")
                out.add(f"{sentence_case(lv)} {sp} across {ts}.")
                out.add(f"{sentence_case(lv)} {sp} through {ts}.")
                out.add(f"{sentence_case(lv)} {sp} near {ts}.")

    inspect_prompts = [
        "Inspect the obstacle before moving.",
        "Check the path before proceeding.",
        "Pause and assess before continuing.",
        "Wait for the path to be clear before moving.",
        "Inspect the surface before stepping forward.",
        "Check for obstacles, then proceed.",
        "Look ahead carefully, then move forward.",
        "Assess the terrain before advancing.",
        "Pause, inspect the passage, then continue.",
        "Verify the path is clear, then proceed.",
        "Inspect the floor before walking.",
        "Stop, check the area, then move slowly.",
    ]
    out.update(inspect_prompts)

    for cond in SOCIAL_CONDS:
        for act in SOCIAL_ACTS:
            out.add(f"{sentence_case(act)} {cond}.")
            out.add(f"When {cond}, {act}.")
            out.add(f"{sentence_case(act)} {cond} and continue only when clear.")

    wait_prompts = [
        "Wait until the path is clear, then move forward.",
        "Hold position until it is safe to proceed.",
        "Pause and wait for further instruction.",
        "Stop and wait if you encounter an obstacle.",
        "Reduce speed and wait near the person.",
        "Slow down and yield to the pedestrian.",
        "Halt and wait before entering the narrow passage.",
        "Stop if the terrain is unsafe.",
        "If uncertain, stop and hold position.",
        "Slow down approaching the uneven surface.",
        "Move carefully and stop if the floor is slippery.",
        "Reduce speed near the crowd and yield.",
    ]
    out.update(wait_prompts)

    return list(out)


def gen_t7() -> list[str]:
    out = set()

    four_step_templates = [
        "Exit the room, turn right, walk forward, and stop at {lm}.",
        "Walk forward, turn left, pass {lm}, and stop near {lm2}.",
        "Crouch under the bar, move forward, stand up, then stop at {lm}.",
        "Walk to {lm}, turn left, go through the corridor, and stop.",
        "Move forward, turn right, follow the corridor, and stop at {lm}.",
        "Step over the obstacle, move to {lm}, turn left, and stop.",
        "Turn around, walk forward, pass {lm}, then stop.",
        "Walk to {lm}, turn right, move forward, then stop near {lm2}.",
        "Go through the corridor, turn left, pass {lm}, and stop.",
        "Turn left, walk to {lm}, turn right, and stop.",
    ]
    five_step_templates = [
        "Exit the room, turn right, follow the corridor, pass {lm}, and stop at {lm2}.",
        "Walk forward, turn left, pass {lm}, enter the corridor, and stop.",
        "Crouch, move forward three steps, stand up, turn right, and stop at {lm}.",
        "Walk forward, duck under the bar, continue, turn right, and stop near {lm}.",
        "Move to {lm}, turn left, pass {lm2}, proceed through the gate, and stop.",
        "Go to {lm}, turn left, walk through the corridor, pass {lm2}, and stop.",
        "Walk forward, sidestep right, pass {lm}, turn left, and stop.",
        "Pass {lm}, turn left, walk through the passage, go to {lm2}, and stop.",
    ]
    six_step_templates = [
        "Exit the room, turn right, follow the corridor, pass the chair, proceed to {lm}, and stop.",
        "Walk forward, turn left, pass {lm}, enter the hallway, go to {lm2}, and stop near the wall.",
        "Crouch under the bar, move forward, stand up, turn right, walk to {lm}, and stop.",
        "Turn right, walk to {lm}, turn left, pass {lm2}, follow the corridor, and stop at the exit.",
        "Move forward, pass {lm}, turn left, walk through the gate, go to {lm2}, and halt.",
        "Enter the corridor, walk forward, turn right, pass {lm}, go to {lm2}, and stop.",
    ]

    all_templates = four_step_templates + five_step_templates + six_step_templates
    for template in all_templates:
        for lm in LANDMARKS:
            for lm2 in LANDMARKS:
                if lm == lm2:
                    continue
                out.add(template.format(lm=lm, lm2=lm2))

    fixed = [
        "Exit the room, turn right, follow the corridor, pass the chair, and stop at the door.",
        "Walk forward, turn left at the marker, pass the table, enter the corridor, and stop near the wall.",
        "Move to the gate, turn right, follow the curved path, pass the cone, and stop at the exit.",
        "Crouch under the bar, move forward, stand up, turn right, walk to the door, and stop.",
        "Back up two steps, turn around, walk forward to the marker, turn left, pass the pillar, and stop.",
        "Walk forward, turn left, pass the table, enter the hallway, go to the gate, and stop.",
        "Move to waypoint A, turn right, pass the cone, go through the corridor, and stop at the wall.",
        "Exit through the door, turn right, follow the marked path, pass the barrier, and stop at the gate.",
        "Walk forward three steps, turn left, pass the chair, proceed to the window, and stop.",
        "Move to the ramp, go up slowly, proceed forward, turn right, and stop at the marker.",
    ]
    out.update(fixed)
    return list(out)


def infer_motion_family(tier: str, prompt: str) -> str:
    p = prompt.lower()
    if tier == "T7":
        return "long_horizon_mixed"
    if any(w in p for w in ["crouch", "kneel", "bow", "lean", "rise", "stand", "squat"]):
        return "posture_transition"
    if any(w in p for w in ["duck under", "step over", "go around", "move past", "obstacle", "bar"]):
        return "obstacle_negotiation"
    if any(w in p for w in ["follow", "path", "waypoint", "route"]):
        return "path_following"
    if any(w in p for w in ["door", "gate", "marker", "wall", "corridor", "table", "chair"]):
        return "landmark_conditioned"
    if any(w in p for w in ["wave", "point", "clap", "nod", "shrug", "salute", "thumbs up"]):
        return "social_gesture"
    if any(w in p for w in ["if", "when", "slow", "carefully", "inspect", "wait", "yield", "blocked"]):
        return "safety_intervention"
    return "basic_locomotion"


def infer_constraint_type(prompt: str) -> str:
    p = prompt.lower()
    if "waypoint" in p:
        return "waypoint"
    if "path" in p or "route" in p:
        return "dense_path"
    if "keyframe" in p or "pose" in p:
        return "keyframe"
    if any(ob in p for ob in ["bar", "gap", "obstacle", "barrier", "debris", "pole", "cable"]):
        return "obstacle"
    if "stop at" in p or "end at" in p or "halt at" in p:
        return "end_pose"
    return "none"


def infer_locomotion_type(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ["wave", "clap", "nod", "salute", "arm", "hand"]):
        return "gesture"
    if any(w in p for w in ["turn", "rotate", "pivot", "spin"]) and not any(
        w in p for w in ["walk", "move", "go", "step", "follow", "pass"]
    ):
        return "turn"
    if any(w in p for w in ["crouch", "kneel", "stand", "rise", "bow", "lean", "squat", "sit down"]):
        return "pose"
    if any(w in p for w in ["and", "then", ","]):
        return "mixed"
    return "walk"


def infer_robot_feasibility(tier: str, prompt: str) -> str:
    p = prompt.lower()
    if tier in {"T6", "T7"}:
        return "needs_review"
    if any(w in p for w in ["obstacle", "bar", "gap", "keyframe", "pose", "corridor", "ramp"]):
        return "needs_review"
    return "safe"


def infer_terrain_dependence(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in [
        "wet floor", "uneven surface", "loose gravel", "slippery tiles",
        "rough ground", "narrow ramp", "steep incline", "unstable surface",
        "corridor", "gate", "door", "marker", "wall", "path", "route",
    ]):
        return "scene_dependent"
    if any(w in p for w in ["obstacle", "bar", "gap", "barrier", "debris", "pole", "cable"]):
        return "mild_obstacle"
    if any(w in p for w in ["forward", "backward", "left", "right"]):
        return "flat_only"
    return "none"


def infer_contact_complexity(tier: str, prompt: str) -> str:
    p = prompt.lower()
    if tier == "T7" or any(w in p for w in ["step over", "duck under", "climb over", "squeeze through"]):
        return "high"
    if tier in {"T4", "T5", "T6"} or any(w in p for w in ["crouch", "kneel", "bow", "turn", "sidestep"]):
        return "medium"
    return "low"


def infer_compositional_structure(tier: str, prompt: str) -> str:
    p = prompt.lower()
    if tier == "T1":
        return "single_action"
    if tier == "T2":
        return "single_action_with_modifier"
    if tier == "T3":
        steps = 3 if p.count(" then ") >= 2 or p.count(",") >= 2 else 2
        return f"{steps}_step_sequence"
    if tier == "T4":
        return "constraint_conditioned"
    if tier == "T5":
        return "landmark_grounded"
    if tier == "T6":
        return "safety_or_social"
    return "long_horizon_sequence"


def infer_ambiguity(prompt: str) -> tuple[bool, str | None]:
    p = prompt.lower()
    if any(w in p for w in ["near ", "beside ", "slightly", "a little"]):
        return True, "relative_spatial_or_extent"
    if any(w in p for w in ["if uncertain", "when the path is clear", "if it is unsafe"]):
        return True, "condition_dependent"
    return False, None


def requires_scene_context(prompt: str) -> bool:
    p = prompt.lower()
    return any(w in p for w in [
        "door", "gate", "marker", "wall", "corridor", "table", "chair",
        "path", "route", "waypoint", "bar", "gap", "ramp", "cone", "barrier",
        "surface", "floor", "person",
    ])


def requires_body_reference(prompt: str) -> bool:
    p = prompt.lower()
    return any(w in p for w in ["arm", "hand", "leg", "body", "chin", "shoulders", "stance"])


def build_constraint_payload(prompt: str) -> dict | None:
    payload = {}
    prompt_lower = prompt.lower()

    path_match = next((pt for pt in PATH_TYPES if pt in prompt_lower), None)
    if path_match:
        payload["path_type"] = path_match

    waypoints = [wp for wp in WAYPOINTS if wp.lower() in prompt_lower]
    if waypoints:
        payload["waypoints"] = waypoints

    obstacles = [ob for ob in OBSTACLES if ob.lower() in prompt_lower]
    if obstacles:
        payload["obstacles"] = obstacles

    landmarks = [lm for lm in LANDMARKS if lm.lower() in prompt_lower]
    if landmarks:
        payload["landmarks"] = landmarks

    return payload or None


def assign_split(idx: int) -> str:
    if idx % 10 == 0:
        return "dev"
    if idx % 10 == 1:
        return "test"
    return "train"


def build_entry(prompt_id: int, tier: str, canonical: str, review_required: bool) -> dict:
    robot_natural = canonical_to_robot_natural(canonical)
    motion_family = infer_motion_family(tier, canonical)
    constraint_type = infer_constraint_type(canonical)
    ambiguity_flag, ambiguity_type = infer_ambiguity(canonical)
    robot_feasibility = infer_robot_feasibility(tier, canonical)

    return {
        "prompt_id": f"kimodo_{prompt_id:06d}",
        "tier": tier,
        "instruction_en": {
            "canonical": canonical,
            "robot_natural": robot_natural,
        },
        "instruction_multilingual": {
            "en": {
                "canonical": canonical,
                "robot_natural": robot_natural,
            },
            **build_translation_stub(),
        },
        "motion_family": motion_family,
        "constraint_type": constraint_type,
        "embodiment": {
            "target_body": "g1",
            "locomotion_type": infer_locomotion_type(canonical),
            "constraint_type": constraint_type,
            "robot_feasibility": robot_feasibility,
            "terrain_dependence": infer_terrain_dependence(canonical),
            "contact_complexity": infer_contact_complexity(tier, canonical),
            "sequence_duration_s": TIER_SPECS[tier]["duration_s"],
            "target_fps": 30,
            "requires_postprocess": tier in {"T4", "T6", "T7"} or constraint_type != "none",
        },
        "kimodo": {
            "validated": False,
            "validation_status": "unverified",
            "review_required": review_required,
            "review_rate": TIER_SPECS[tier]["review_rate"],
            "motion_asset": None,
            "constraint_payload": build_constraint_payload(canonical),
        },
        "metadata": {
            "qc_focus": TIER_SPECS[tier]["qc_focus"],
            "compositional_structure": infer_compositional_structure(tier, canonical),
            "ambiguity_flag": ambiguity_flag,
            "ambiguity_type": ambiguity_type,
            "requires_scene_context": requires_scene_context(canonical),
            "requires_body_reference": requires_body_reference(canonical),
            "translation_status": "pending",
            "safe_for_direct_execution": robot_feasibility == "safe" and not ambiguity_flag,
            "validation_status": "unverified",
            "split": assign_split(prompt_id),
        },
    }


def main() -> list[dict]:
    out_dir = Path("corpus_build")
    out_dir.mkdir(exist_ok=True)

    generators = {
        "T1": gen_t1,
        "T2": gen_t2,
        "T3": gen_t3,
        "T4": gen_t4,
        "T5": gen_t5,
        "T6": gen_t6,
        "T7": gen_t7,
    }

    all_prompts = []
    prompt_id = 0
    retained_candidate_total = 0

    for tier, gen_fn in generators.items():
        spec = TIER_SPECS[tier]
        raw_candidates = dedupe_casefold(gen_fn())
        candidate_pool = constrain_pool(raw_candidates, spec["candidate_quota"])
        retained_candidate_total += len(candidate_pool)

        selected = candidate_pool[:spec["target"]]
        shortfall = spec["target"] - len(selected)
        status = "✓" if shortfall == 0 else f"⚠ SHORT by {shortfall}"
        print(
            f"{tier}: {len(raw_candidates):5d} generated → "
            f"{len(candidate_pool):5d} review-pool → "
            f"{len(selected):5d} selected  {status}"
        )

        review_indices = sample_review_indices(len(selected), spec["review_rate"])
        for idx, prompt in enumerate(selected):
            all_prompts.append(build_entry(
                prompt_id=prompt_id,
                tier=tier,
                canonical=prompt,
                review_required=idx in review_indices,
            ))
            prompt_id += 1

    out_path = out_dir / "corpus_raw.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for entry in all_prompts:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nRetained English candidate pool : {retained_candidate_total}")
    print(f"Final selected prompts         : {len(all_prompts)}")
    print(f"Saved to                       {out_path}")
    return all_prompts


if __name__ == "__main__":
    main()
