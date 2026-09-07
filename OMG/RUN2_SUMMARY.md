# Run 2 — what we did, what happened, what next

*2026-09-07. Plain-language summary. Full detail and provenance live in `MAILA_SSOT_2026-09-05.md`.*

---

## What we built

OMG is a frozen humanoid-motion model that turns **English** text into Unitree G1 motion. We
bolted a small **adapter** onto it so it takes **Hindi** instead. Everything else is frozen —
the 100M-parameter motion model, MuRIL, t5. We trained **4.7M parameters**, about 4% the size
of the model we are steering, on one rented GPU for ~7 hours.

## How do we know if it works?

One number carries most of this: **caption sensitivity**.

Give the model the *right* caption, measure how well it predicts the motion. Then give it a
*wrong* caption. If it is genuinely reading the text, the wrong caption should hurt.

- **0%** = the model is deaf to the caption
- **higher** = the caption is really steering the motion

## The headline result

All measured on the test set, same protocol for every row:

| what | caption sensitivity |
|---|---|
| no text at all (floor) | 0.00% |
| **stock OMG given Hindi** — what you get today | **+1.80%** |
| stock OMG given English (zero-shot) | +9.80% |
| **our adapter given Hindi** | **+10.46%** |
| our adapter given English (control) | +14.69% |

**Hand stock OMG a Hindi caption and it is effectively deaf — 1.80%.** t5 was never trained on
Devanagari, so different Hindi sentences collapse to nearly the same internal vector.

**Our adapter takes that to 10.46% — a 5.8× lift**, and 17× on a stricter variant of the same
test. Reading Hindi, it follows the caption slightly *better* than the original model does
reading English.

## But: 71%, not parity

Against an English adapter trained **identically** on the same data, Hindi reaches **71%**
(10.46 vs 14.69). There is a real gap, and it has measured causes:

- MuRIL needs **1.16–1.28× more tokens** for Indic scripts against a hard 50-token window
- the Hindi captions are machine-translated, which collapses distinct English into identical
  Hindi strings

An earlier version of this summary said "104% of English." That compared against **zero-shot**
t5, which never saw our data — an unfair baseline. The English control arm is what corrected it.

## Motion quality

From 6 full-length (8–10 s) closed-loop renders against real source mocap:

| metric | stock+en | stock+hi | ours+en | ours+hi |
|---|---|---|---|---|
| MPJPE (mm) ↓ | 143.5 | 151.5 | **129.8** | 139.2 |
| joint-limit rate ↓ | 0.587 | 0.582 | 0.379 | **0.124** |
| joint hard-limit ↓ | 0.120 | 0.210 | 0.090 | **0.008** |
| foot sliding ↓ | 0.032 | **0.028** | 0.144 | 0.180 |

**Both adapter arms beat both stock arms on pose accuracy**, and our Hindi arm (139.2 mm) beats
stock OMG on *English* (143.5 mm). On joint-limit legality the Hindi arm is dramatically best —
**0.8% hard violations against 12–21% for stock**, a 15–26× reduction.

Read the "stock+hi wins" cells carefully. Stock OMG on Hindi is near text-blind, so it barely
moves — on one clip it travelled 0.01 m where ground truth travelled 1.46 m. A robot standing
still cannot slide its feet. Those metrics reward inaction, so treat them as an artifact.

---

## What went well

- **It works, and we know why.** Two structural bugs were the whole story: the adapter's output
  was 4× too large for the frozen model to read, and then every caption produced nearly the same
  vector (cosine 0.9966 between different captions). Before fixing them, sensitivity sat at ~0%
  no matter how long we trained. That is a real diagnosis, not a hyperparameter search.
- **Cheap.** 4.7M trainable parameters, ~7 GPU-hours, ~$5 of compute.
- **The weak English anchor won.** λ_en 0.1 and 0.5 came out statistically tied, meaning the
  motion-pivot objective carries the signal rather than English distillation doing the work.

## What did not

- **The corpus was too small.** 12,951 clips. Validation peaked at step 20,000 then *declined* —
  the signature of running out of data, not out of model capacity.
- **Hindi only.** The actual research question is multilingual; Run 2 tested one language.
- **Four measurement bugs**, each caught only by cross-checking. Several numbers were reported
  wrong before correction, including a whole conclusion that had to be withdrawn. Every headline
  figure here has since been re-measured under a verified protocol.
- **n = 6 clips** for motion quality. Suggestive, not significant. The joint-limit result (26×)
  is the only one robust enough to defend as-is.

---

## Next target — Run 3

| | Run 2 | Run 3 |
|---|---|---|
| languages | Hindi | **Hindi, Bengali, Tamil, Telugu** |
| motions | 12,951 | **44,996 episodes / 32,519 motions** |
| corpus | AMASS/HumanML3D | OMG-Data (natively G1, no retargeting) |
| adapter | one per language | **one shared across four** |

The corpus is extracted and verified. Translation is running across three machines. The English
control arm is in the plan from the start this time, not added afterwards.

---

## On Contribution 2 (mirror counterfactuals) — yes, but with one condition

**The idea:** the dataset contains mirrored motion pairs. Swap "right"↔"left" in the caption and
the motion should flip. Penalise the model when it does not. This targets a weakness we have now
measured **twice** — Run 1's worst failure was a direction word ("steps to their right":
English 1.66 m, Hindi 0.06 m), and Run 2's global-trajectory gap points the same way.

**Use it.** ~22.6% of captions carry exactly one side word, giving **~10,000 counterfactual
pairs** in the Run 3 corpus — 18× what the evaluation needs.

**Three conditions, all non-negotiable:**

1. **Mirrors are paired supervision, never extra training rows.** As bulk augmentation they
   would (a) defeat our leakage check silently, since a mirror shares neither source id nor
   caption with its original, (b) turn the flip-rate metric into a memorisation test, and
   (c) reduce the contribution to "we added flipped data." The split builder now has a
   `mirror::` key for exactly this; it is tested and it fails correctly when removed.
2. **Run the English adapter with identical counterfactual supervision.** Without it the
   obvious review attack lands: *"your adapter got this supervision, frozen English did not."*
   Against frozen English alone (~4% flip rate) the "surpasses English" claim does not survive.
3. **Hold out pairs for evaluation** whose source motion never appeared in training.

**Both outcomes are worth publishing.** If Indic ≥ English under identical supervision, that is
the strong claim and it is defensible. If both land far above 4% and near each other, the finding
becomes "attribute grounding is learnable in the adapter regardless of source language, and the
deficiency is in the frozen backbone" — weaker, but robust, and it still fixes a measured flaw in
a widely-used model.

Only the second is reachable without the control arm, and only if you run it.
