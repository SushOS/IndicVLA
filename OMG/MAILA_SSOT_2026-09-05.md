# MAILA — Single Source of Truth (2026-09-05)

**Status:** This document supersedes all prior planning documents as the reference for the work
going forward. It is not a replacement for `MAILA_RESEARCH_LOG_2026-08-22.md` — that log remains
the authoritative record of **Run 1** and its artifacts, and is cited throughout. This document
records: what Run 1 established, why the approach changed, what the new approach *is* and why it
was chosen over alternatives, the corpus decision and every measurement behind it, and the plan
for Run 2.

**Standing epistemic rule, inherited and reaffirmed:** *the PDF is not the source of truth; the
code and the actual run artifacts are.* Every number below is tagged with how it was obtained:

| tag | meaning |
|---|---|
| **[M]** | Measured by us, reproducible from a named artifact or a named script |
| **[C]** | Read directly from source code, cited by file |
| **[P]** | Published claim from a paper, cited by arXiv ID — a claim, not a verified fact |
| **[E]** | Estimated/derived, with the assumption stated |
| **[?]** | Unverified. Flagged deliberately so nobody treats it as settled |

---

## Contents

1. [The goal](#1-the-goal)
2. [The frozen foundation: OMG](#2-the-frozen-foundation-omg)
3. [Run 1 — what it established, and what it did not](#3-run-1--what-it-established-and-what-it-did-not)
4. [Why the approach changed](#4-why-the-approach-changed)
5. [The landscape (deep-research pass, 2026-08-30)](#5-the-landscape-deep-research-pass-2026-08-30)
6. [The new approach: motion-pivot cross-lingual training](#6-the-new-approach-motion-pivot-cross-lingual-training)
7. [The collapse analysis — the central risk](#7-the-collapse-analysis--the-central-risk)
8. [Contributions 2 and 3](#8-contributions-2-and-3)
9. [The corpus decision — the full search](#9-the-corpus-decision--the-full-search)
10. [The data as built](#10-the-data-as-built)
11. [Run 2 experiment plan](#11-run-2-experiment-plan)
12. [Metrics and diagnostics](#12-metrics-and-diagnostics)
13. [Open questions and risks](#13-open-questions-and-risks)
14. [Errors and lessons](#14-errors-and-lessons)
15. [Artifact and reference manifest](#15-artifact-and-reference-manifest)

---

## 1. The goal

Accept a sentence in an Indic language — Hindi, Bengali, Tamil, Telugu — and produce Unitree G1
whole-body motion that a physics tracker can execute, **with no translation step at inference**.

The end state is a research contribution defensible at an A\* venue. The system exists; Run 1
proved that. What this document plans is the work that makes it *novel* rather than merely
*working*.

**A naming discipline, adopted deliberately:** this is **text-to-motion for humanoid control**, not
a VLA. There is no vision input. Calling it a VLA in a paper invites a reviewer objection we would
lose. The repository name `IndicVLA` is historical; the work is MAILA.

---

## 2. The frozen foundation: OMG

**OMG: Omni-Modal Motion Generation for Generalist Humanoid Control**, arXiv:2606.10340
(Tsinghua MARS Lab). Code `github.com/Tsinghua-MARS-Lab/OMG`, weights `THU-MARS/OMG`, data
`THU-MARS/OMG-Data`.

Its thesis is a **generator–tracker hierarchy**: a diffusion "brain" that reasons over multi-modal
conditions, atop a reactive tracking "cerebellum" (HoloMotion) that executes on the robot. **MAILA
operates entirely at the brain layer.** The tracker is never touched.

### 2.1 The contracts that bind our work

| Property | Value | Source |
|---|---|---|
| Checkpoint | `updated/100m/sstep=170000.ckpt` | **[M]** |
| Backbone | hidden 768, 10 layers, 12 heads, mlp_ratio 4.0, text_dim 768 | **[C]** `configs/generation/denoiser/transformer_100m.yaml` |
| Motion representation | 125-D = root_pos(3) + root_rot_6d(6) + joints(29) + links(87) | **[C]** `src/omg/motion/feature_codec.py` |
| History / horizon / rate | L=10, H=60, 30 FPS | **[C]** |
| Diffusion | 1000 steps, cosine β, `pred_x0` | **[C]** `configs/generation/diffusion/guided_x0.yaml` |
| Sampler | DDIM 50 steps, η=0, CFG 2.5 | **[M]** |
| Text interface | T5-base, **max_length=50**, width 768 | **[C]** `src/omg/generation/conditions/t5.py` |
| `text_mask_prob` in OMG's own training | 0.3 | **[C]** |
| **qk-norm contract for `updated/100m`** | **self=true, cross=true** | **[M]** EXP-F |

### 2.2 Three facts that are architecturally load-bearing

1. **`self.proj` is `nn.Identity()`** **[C]** — because t5-base's `d_model` equals the DiT's
   `output_dim` (both 768). There is **no learned layer anywhere between the text encoder and
   cross-attention**. Any replacement encoder must land in t5-base's *actual output geometry*, not
   merely in some 768-dimensional space. **A shape match is not a distribution match.** This is the
   project's most recurrent failure class.

2. **Padding is `padding="max_length"`** **[C]**, so a valid mask is always a contiguous prefix.

3. **The qk-norm setting is invisible to `state_dict` loading** **[M]** — it changes attention
   numerics without changing any parameter shape, so `load_state_dict(strict=True)` cannot detect a
   mismatch. It had to be *measured*, and it differs per checkpoint family: `true/true` wins by
   **+16.1 GEN-R@1 points** on `updated/100m`, whereas `none` had won for `paper/300m`. Gate A
   (REFERENCE R@1 = 0.6680, checkpoint-independent) passed exactly in both arms, validating the
   measurement stack before the generated numbers were trusted.

| EXP-F arm | self_qk | cross_qk | REFERENCE R@1 | GENERATED R@1 | FID |
|---|:--:|:--:|---|---|---:|
| `none` | false | false | 0.6680 | 0.4971 | 0.3211 |
| **`self_and_cross`** | **true** | **true** | 0.6680 | **0.6582** | 0.2391 |

### 2.3 The evaluator

Text-to-motion metrics come from a separately-trained TMR-style model, `evaluator/step_004000.pt`
**[M]**: a `body_pos_local` RoPE transformer on the motion side (dim 512, frozen), and a **trained
`Linear(1024,512)+LayerNorm`** projection on a **frozen t5-3b** on the text side. R-Precision uses
`TEXT_RETRIEVAL_BATCH_SIZE = 32` **[C]**, stratified by source dataset; **chance R@1 = 3.125%**.

**The evaluator cannot read Devanagari** — t5's tokenizer produces 48–49% `<unk>` on Devanagari
**[M]**. Every Hindi retrieval number we report is therefore scored against the *English* caption
for the same motion. This is a **measured lower bound**, not a Hindi semantic score, and §12
carries the plan to fix it.

---

## 3. Run 1 — what it established, and what it did not

Run id `maila-hi-20260822-100455`. Full detail in `MAILA_RESEARCH_LOG_2026-08-22.md`.

### 3.1 The system

Frozen MuRIL → trainable residual adapter → frozen OMG-100M → 60-frame 125-D G1 motion.
**4,725,505 trainable parameters in 9 tensors**, asserted at every model build **[C]**
`maila_train.py::build`.

```
U = LayerNorm(H)                       # H: (B,50,768) frozen MuRIL states
R = Linear(3072,768)(Dropout(GELU(Linear(768,3072)(U))))
Z = OutputLayerNorm(U + alpha * R)     # alpha learnable, init 0.1
```

Objective: `L_total = L_native + λ_resp(step)·L_response`, where `L_response` matches the
student's `pred_x0` to a **frozen English T5 teacher's** `pred_x0` under a **replayed RNG state**
so that τ, ε and the noisy input are identical between branches and the *only* difference is the
conditioning text.

Schedule: `λ_resp` 0.5 held to step 5,000 then linear decay to 0.1 by 25,000; teacher on 50% of
minibatches; `text_mask_prob = 0.0` (deviation from OMG's 0.3, justified: with a frozen DiT and a
cached t5 null, a dropped-text row yields **zero adapter gradient**).

Training: 25,000 steps, batch 64 × accum 4 = effective 256, lr 1e-4 with 1,000-step warmup and
cosine decay to 1e-6, AdamW wd 0.01, grad clip 1.0, bf16. **0.393 s/step, 2.73 h wall clock** on one
RTX 4090 **[M]**.

### 3.2 Results

**Checkpoint selection** required a second controlled pass: in-loop validation oscillated in a
0.029–0.046 band with no trend because it sampled fresh timesteps each eval. `finalize_run.py`
re-scored all 13 checkpoints on **768 fixed-seed val windows** with per-batch seeds replayed
identically, selecting on text-sensitivity rather than native loss:

```
rel = (L_wrong − L_correct) / L_correct
```

| checkpoint | L_correct | L_wrong | rel | % of RANDOM→ENGLISH gap |
|---|---:|---:|---:|---:|
| RANDOM (untrained) | 0.10313 | 0.10193 | **−1.17%** | 0.0% |
| ENGLISH_t5 (ceiling) | 0.04757 | 0.06122 | **+28.68%** | 100.0% |
| **step 22000 — SELECTED** | 0.04815 | 0.05900 | **+22.54%** | 79.4% |

Steps 20k/22k/24k/25k sit within a 1-point band; the supported claim is *"converged in the 20k–25k
range,"* not *"22000 is best."*

**The 3-arm benchmark** (n=1024 test windows, step 22000, CFG 2.5, DDIM-50) **[M]**:

| arm | R@1 | Matching ↓ | Diversity | FID ↓ | KID ↓ |
|---|---:|---:|---:|---:|---:|
| REFERENCE (real motion, real caption) | 0.6416 | 1.0767 | 1.3214 | — | — |
| **ENGLISH** (frozen t5) | **0.6670** | 1.0692 | 1.3055 | 0.0593 | 3.20e-05 |
| **HINDI_ADAPTER** | **0.5020** | 1.1115 | 1.3036 | **0.0631** | 4.76e-05 |
| **MT_PIVOT** (IndicTrans2 → t5) | **0.5352** | 1.1507 | 1.3554 | 0.1502 | 2.84e-04 |

The adapter retains **75.3%** of English R@1; MT retains **80.2%**. **Translation wins retrieval by
3.3 points; the adapter wins distributional realism by 2.4×** (FID 0.0631 vs 0.1502, KID agreeing in
rank). This independently reproduces PEA-Diffusion's qualitative finding (arXiv:2311.17086) in a new
modality and on a different frozen backbone.

**Direction minimal pairs** (n=100 per arm, matched seed pose and noise, only the direction word
swapped) **[M]**: English flip rate **4.0%**, Hindi **10.0%**. The base checkpoint is essentially
**direction-blind in English too**. Underpowered (≈550 pairs/arm needed for significance), so the
defensible claim is narrow: *there is no large Hindi-specific direction deficit.*

### 3.3 What Run 1 did **not** establish

- **Nothing about native Hindi.** Every caption on both sides of every comparison was
  machine-translated. Per Artetxe et al. (EMNLP 2020), translated evaluation data can produce false
  improvements from induced artifacts.
- **Nothing about any language other than Hindi.**
- **Nothing that beats translation on the primary metric.** §7.3 of the Run-1 log is explicit: *"the
  adapter beats translation" is not supported;* "comparable semantic transfer, better motion realism,
  no MT dependency" is.

---

## 4. Why the approach changed

Run 1's design — a frozen encoder, a small trainable adapter, a frozen generative backbone — is a
**known and published pattern**: AltDiffusion (arXiv:2308.09991, AAAI 2024), PEA-Diffusion
(arXiv:2311.17086, ECCV 2024), MuLan (arXiv:2412.01271). Porting it from images to motion, for one
language, and reproducing the field's existing qualitative finding is **solid, incremental work**.
It is not an A\* contribution.

Three limits fall directly out of the objective's *shape*:

1. **English is privileged.** Every other language is defined as a deviation from English. A fifth
   language learns to imitate English again, separately, from scratch.
2. **Languages never meet.** A Hindi adapter and a Tamil adapter never exchange a gradient.
3. **It caps at the teacher.** Asymmetric distillation says *be as good as English*. It cannot say
   *be better than English* at what English is bad at — such as left/right, where the frozen backbone
   sits at a 4% flip rate.

The insight that reframes the work: **the teacher was never the point — the pivot was.**

---

## 5. The landscape (deep-research pass, 2026-08-30)

Full record: `DEEP_RESEARCH_PROMPT_2026-08-30.md` (brief) and
`DEEP_RESEARCH_FINDINGS_2026-08-30.md` (verdicts).

**Language→humanoid-motion is a crowded English field.** All **[P]**: UH-1/Humanoid-X
(arXiv:2412.14172 — VQ action tokens, codebook 2048, CLIP text encoder, causal AR transformer,
Unitree H1-2, 20M poses from 160K videos); Humanoid-LLA (arXiv:2511.22963 — pretrained LLM with a
unified motion vocabulary, motion chain-of-thought + RL from physical feedback, Booster T1, code
unreleased); FRoM-W1 (arXiv:2601.12799); TextOp (arXiv:2602.07439 — **streaming** text commands,
real-time, on-the-fly modification); UniAct (arXiv:2512.24321); LeVERB (arXiv:2506.13751); DAJI
(arXiv:2605.14417 — anticipatory contact/balance intent from *full* commands); SafeFlow
(arXiv:2603.23983); ECHO (arXiv:2603.16188); RoboForge (arXiv:2603.17927); RLPF (arXiv:2506.12769).
Graphics-side motion-LLMs are equally crowded (MotionGPT lineage, IRG-MotionLLM arXiv:2512.10730,
OmniMoGen arXiv:2512.19159).

**Every one is English-conditioned.**

**The white space is documented by third parties.** Two 2026 analyses **[P]** — *Beyond English*
(arXiv:2606.15714) and *When Does Language Matter?* (arXiv:2606.11906) — measure a large multilingual
gap in VLA models and show it **persists even when the underlying LLM backbone is multilingual**.
Neither proposes a fix. Someone else has already proven that the cheap alternative — prompt the
English system in Hindi — fails.

**Multilingual motion generation has exactly one prior work:** BiMD / BiHumanML3D
(arXiv:2603.25178) **[P]** — English–Chinese, 13,312 motions, built by DeepSeek → Qwen refinement →
**17 human annotators verifying 53%** (91.3% inter-annotator agreement). Its method, Cross-Lingual
Alignment (CLA), aligns **text embeddings** via KD from an English-centric vision-language teacher
(OpenCLIP), inside MLD latent diffusion in SMPL graphics space. Code-switching is shown as a
qualitative figure, not a measured axis.

**No Indic text-to-motion, no Indic embodied instruction-following, and no non-English humanoid
whole-body control exists on current evidence.**

**Consequences for strategy.** "First LLM with a motion vocabulary for a humanoid" is dead
(Humanoid-LLA). "Streaming humanoid control" is dead (TextOp). "Anticipatory physical intent" is
dead (DAJI). What survives is the *language* axis — and the gap papers are the published
justification that it matters.

---

## 6. The new approach: motion-pivot cross-lingual training

### 6.1 The idea

Ask what is genuinely shared between a Hindi sentence and a Tamil sentence describing the same clip.
Not the words. Not word order. Not the tokens — MuRIL cuts Devanagari and Tamil script into different
numbers of pieces at different boundaries. Not the embedding vectors, which only look comparable
because they share a width of 768.

**What is shared is the motion.** A 60×125 tensor of joint angles and link positions, identical
regardless of which language asked for it. Not a learned latent that might drift — the physical
object the system exists to produce, renderable, trackable, and falsifiable.

> **Two sentences in different languages count as equivalent when, holding the motion, the history,
> the diffusion timestep and the noise all fixed, the frozen generator predicts the same clean
> motion from both.**

BiMD puts the equals sign **before** the generator, in text space, and therefore needs a teacher and
sentence-aligned parallel text. We put it **after** the frozen generator, on the predicted motion.
Nothing between the captions and the motion is constrained: the adapter may represent Hindi and Tamil
however it likes, provided the frozen generator does the same thing with both.

Because the comparison happens *after* cross-attention has consumed the context — and cross-attention
is permutation-invariant over context tokens — word order washes out on its own and padding
differences are absorbed by the mask. **We compare the functional effect of two contexts, not their
geometry.** That is why it survives four scripts with four tokenizers.

### 6.2 The objective

```
L = ½[ L_native(A) + L_native(B) ]
  + λ_x(step) · ‖ x̂₀(A) − x̂₀(B) ‖²          cross-lingual consistency
  + λ_en      · ‖ sg[x̂₀(en)] − x̂₀(A) ‖²      English geometric anchor
```

- `L_native` is **OMG's own** `diffusion.training_losses` — called, not reimplemented, so the
  reduction matches the pipeline validated against a published reference number.
- Both Indic branches **carry gradient**. This is a sibling relationship, not teacher–student.
- `λ_en` is retained deliberately: English is no longer the semantic authority, but **t5-base's
  output geometry is still the space the frozen cross-attention was actually fitted in.**

### 6.3 One training step

1. Draw the batch once — ground-truth future `x₀`, history `h`. Nothing language-specific yet.
2. Sample an unordered language pair from {hi, bn, ta, te}. Six pairs, all covered over training.
   Two branches, not four, keeps step cost near 2×.
3. Save the RNG state (`torch.cuda.get_rng_state()` + `torch.get_rng_state()`).
4. Run branch A: caption in language A → frozen MuRIL → adapter → frozen DiT. Keep `pred_x0`.
5. **Restore the RNG state.** The next branch now draws the same τ, the same ε — and the same
   adapter dropout mask, because tensor shapes do not depend on the caption.
6. Run branch B with language B. Carries gradient too.
7. Combine per §6.2.

**Why the RNG replay is load-bearing.** With independent noise, the difference between branches
would be dominated by having started at different points in the diffusion process. We would be
measuring variance and calling it semantics, and the gradient would be noise chasing noise. Replaying
turns the pair into a controlled experiment: two conditions, one manipulated variable.

**The RNG order is provably preserved** **[C]**: inside `native_loss` the conditions are built first
(adapter dropout), then `training_losses` samples τ and ε. Both draws have caption-independent
shapes — dropout is `[B,50,3072]` whatever the sentence, and the mask changes only which positions
are *valid*, not how many random numbers are consumed.

### 6.4 The code delta against Run 1

Three edits to `maila_train.py`:

```python
# --- Run 1: Hindi student, frozen English teacher, one direction ---
state = save_rng()
l_native, diff, target, valid, hlen = native_loss(model, batch)
restore_rng(state)
with torch.no_grad():                                 # <-- EDIT 1: delete
    enc  = teacher(batch["caption_en"])               # <-- EDIT 2: delete override
    d_en = model.diffusion.training_losses(model.denoiser, target, conds_en, valid, history_len=hlen)
l_resp = F.mse_loss(diff["pred_x0"], d_en["pred_x0"].detach())

# --- Run 2: two Indic languages through the SAME adapter ---
la, lb = rng.choice(LANGS, size=2, replace=False)     # <-- EDIT 3: language column
state = save_rng()
batch["caption"] = batch[f"caption_{la}"]
l_nat_a, diff_a, target, valid, hlen = native_loss(model, batch)
restore_rng(state)                                    # SAME tau, eps, x_t, dropout
batch["caption"] = batch[f"caption_{lb}"]
l_nat_b, diff_b, *_ = native_loss(model, batch)       # carries gradient
l_xling = F.mse_loss(diff_a["pred_x0"], diff_b["pred_x0"])   # no detach
loss = 0.5*(l_nat_a + l_nat_b) + lam_x(step)*l_xling + lam_en*l_en_anchor
```

**The code delta is small. The claim delta is not.** State this honestly in the paper.

### 6.5 What actually changes, precisely

| Axis | Run 1 | Run 2 |
|---|---|---|
| Branch-2 encoder | frozen T5 — a *different* network | same MuRIL + same adapter |
| Branch-2 gradient | `no_grad` + `.detach()` | full gradient, symmetric |
| **What the target is** | **English's prediction of the motion** | **another language's prediction; both pulled to x₀** |
| Languages in the adapter | Hindi only | hi + bn + ta + te, shared weights |
| Optimization character | distillation, **fixed** target | joint alignment, **moving** target |
| Failure mode | under-fitting the teacher | **mutual collapse** — needs a detector |
| Ceiling | English, by objective and by reporting | none built in |
| Zero-shot question | not askable | the headline result |

**The deepest difference.** Run 1's response term regresses toward `d_en["pred_x0"]` — the *frozen
English pipeline's prediction*, not ground-truth motion. Ground truth enters only through
`L_native`. So Run 1 decomposes as: `L_native` → anchors to the real motion; `L_response` → anchors
to *English's opinion about* the real motion. That is precisely PEA-Diffusion's shape. In Run 2 both
branches regress toward `x₀` through their own native losses and the consistency term ties their
errors together. **The difference between *be like English* and *agree with each other about the
physics*.**

### 6.6 Practical notes

- **Memory.** Two backward passes now traverse the frozen DiT (frozen but differentiable — the only
  route to the adapter). On a 24 GB 4090, use micro-batch 32 with accumulation 8 to hold effective
  batch 256. Expect ~1.5–1.8× the 0.393 s/step of Run 1 **[E]**.
- **Timestep weighting.** At very high τ the input is near-pure noise and both branches fall back on
  the prior, agreeing for reasons unrelated to language; at very low τ the prediction is dominated by
  `x^τ`. The informative band is in the middle. If the consistency term looks suspiciously easy,
  weight it by `ᾱ_τ` or restrict it to a τ window — **and report that you did.**

---

## 7. The collapse analysis — the central risk

### 7.1 The sign flip

Run 1's English target is **caption-dependent**, so matching it *forces* caption-dependence. **The
English teacher was an active anti-collapse force.** The consistency term has the opposite sign: the
cheapest way for two branches to agree is for neither to read its text. Removing one and adding the
other changes the pressure **twice, in the same direction.**

### 7.2 Does collapse pay?

From the selection table **[M]**: getting the caption right rather than wrong is worth
`0.05900 − 0.04815 = 0.01085` — about **18% of the loss level** (English ceiling: 22%). Which states
the uncomfortable thing plainly: **~80% of the diffusion loss is explained by history, noise and the
generic motion prior; text is worth ~20%.**

Against that, Phase-1 measured `mean|Δx₀| = 5.157e-02` between different prompts, putting
`L_xling` near `4.2e-3` at initialization **[E]**; at `λ_x = 0.5` the prize for collapsing caps at
`~2.1e-3`.

> **Collapse costs ≈ 0.011 and saves ≈ 0.002 — a ~5× losing trade at λ_x = 0.5.
> Break-even sits near λ_x ≈ 2.6.**

**Therefore: `λ_x ≤ 0.5`, never approaching 1.0.** That is a margin, not a taste.

**Caveat that must travel with those numbers [?]:** `L_wrong` is *text present but wrong*, not *text
absent*. A wrong caption actively steers toward a different motion, which may be worse than no
caption. **`L_null` has never been measured, and its direction relative to `L_wrong` is unknown.**
The 5× figure rests on substituting one for the other. **Measuring `L_null` is the first task of
Run 2** (§11.0).

### 7.3 The failure to actually fear is partial

Full text-blindness is economically irrational. **Selective deafness is not.** Most of the 0.011
reward comes from coarse action identity. Fine attributes — left/right, speed, count — move a small
fraction of the 60×125 tensor, so discarding *them* costs almost nothing on native loss while paying
full price on consistency, because fine attributes are where four languages diverge most in surface
form. The gradient's incentive: **keep the verb, discard the adverb.** Aggregate loss, `rel` and R@1
will all look fine while it happens.

Corroboration: the frozen backbone is **already** history-dominated. The English 4% direction flip
rate means 96% of the time swapping left for right does not change where the robot goes; the
side-by-side set's Spearman +0.75 / Pearson +0.80 between English and Hindi displacement is what you
would expect if both arms were largely driven by shared history.

### 7.4 Guards, ranked

1. **`λ_en = 0.1` as DEFAULT, not ablation.** It restores exactly the pressure being removed.
2. **A language-symmetric repulsive term, if needed.** Roll captions by one within the batch so item
   *i* gets item *i+1*'s caption **with its own history and noise**:
   `L_repel = max(0, m − mean_i‖x̂₀(c_i) − x̂₀(c_{i+1})‖²)`. One extra forward yields B negatives
   (2 → 3 branches, ~1.5×). Unlike consistency alone, **this has no collapse solution**.
3. **Log text-induced variance every validation as a pure diagnostic** — the inner quantity above,
   per language, never optimized, so it stays a valid selection signal and moves long before `rel`
   or R@1 does.

> **Methodological trap:** if you train on `L_repel`, the caption-swap metric can no longer select
> checkpoints — it becomes a training objective reported back to itself. Select on the **val
> direction minimal-pair flip rate** instead, which is differently constructed and more meaningful.

---

## 8. Contributions 2 and 3

### 8.1 Contribution 2 — mirror-counterfactual attribute grounding

**This is not a separate contribution sitting beside Contribution 1. It is the structural antidote to
Contribution 1's failure mode**, applying targeted supervision to exactly the fine-attribute subspace
that cross-lingual consistency is incentivized to erase, using ground truth that is *physically
exact*. **Run them together, not sequentially.**

Mechanism: mirror-paired motion with side-word-swapped captions, penalizing the model unless
swapping दाहिने↔बाएं flips the predicted lateral offset. Train into the adapter first; if the frozen
prior cannot express it, escalate to the rank-8 K/V cross-attention LoRA the original plan already
sanctions as the targeted fallback, keeping all temporal layers frozen.

**Open build item (§13.1):** the chosen corpus has **no mirrors**. See §10.4.

### 8.2 Contribution 3 — IndicMotion-G1 benchmark

- Native-**verified** test sets in all four languages (verification+correction is far cheaper than
  authoring; BiMD's 53%-verified / 91.3%-agreement is the bar), plus a native-**authored** gold
  subset from the untouched `reserve` split.
- **Romanized and code-mixed axes** — where translation structurally cannot compete, since there is
  no single source language to translate *from*. MuRIL's transliteration pretraining
  (arXiv:2103.10730) is an asset never yet exercised.
- **Attribute minimal pairs** with rule-based kinematic scoring.
- **A MuRIL-based multilingual evaluator** to replace the English-mediated scoring of §2.3.
- **Physical/tracker metrics** — what makes it the first multilingual benchmark for *robot-executable*
  motion. BiMD lives in SMPL graphics space: no falls, no joint limits, no tracker.

---

## 9. The corpus decision — the full search

Run 1 used BONES-SEED. Measuring it against the official source exposed a structural problem, and the
search for a replacement was resolved by measurement, not preference.

### 9.1 BONES-SEED is not viable for this work

Downloaded `bones-studio/seed` → `metadata/seed_metadata_v004.csv` (146 MB, 142,220 rows, 51
columns — our local copy is this plus 16 translation columns) **[M]**:

| | |
|---|---:|
| Clips | 142,220 |
| Distinct `desc_1` captions | **5,272** |
| **Clips per caption** | **27.0** |
| Clips whose caption is unique to them | **14 (0.010%)** |
| 229 captions (4.34%) cover | **50% of all clips** |
| `Baseline` category | 22,878 clips (16.09%) from 26 concepts, **15 captions** |

The cause is structural, not a labelling defect: the caption is a property of the **concept**, not
the performance (`content_names with >1 distinct desc_1 = 0`), and 522 actors — mean 8.5, max 400 per
concept — each perform it. Mirroring is 50/50 (71,132 / 71,088) and mirrors inherit the caption.

This **independently verified all four of the Run-1 log's §4.1 claims exactly**: 7,620 content_names,
5,272 captions, 2,338 captions spanning 4,686 content_names, Baseline 16.09% from 26 concepts at 880
clips each.

Two further findings: **`desc_4` is a different kind of field** — 17,754 distinct (3.4× `desc_1`),
varying within 99.1% of concepts (mean 2.99, max 6), i.e. *performance-level* rather than
*concept-level*, while `desc_1/2/3` (5,272/5,365/5,430) are all concept-level. And **machine
translation collapsed 12 English captions across 149 train clips**, systematically on gait: *"stop
walking forward"* and *"stop jogging forward"* both map to `सामान्य गति से आगे बढ़ना बंद करें`. This
also resolves the Run-1 log's unreconciled 4,038 → 4,015 caption discrepancy: 4,038 English cap_keys
→ 4,026 after MT collapse **[M]** → 4,015 at window level after short-clip drops **[E]**.

### 9.2 The alternatives, measured

All measured by downloading the actual annotation files and tokenizing with t5-base **[M]**:

| Dataset | captions | distinct | **clips/cap** | used once | **t5 median** | **>50 tok** |
|---|---:|---:|---:|---:|---:|---:|
| **HumanML3D** (originals) | 43,692 | 40,087 | **1.09** | 96.6% | **15** | **0.7%** |
| HumanML3D (+mirrors) | 87,384 | 55,408 | 1.58 | 54.7% | 15 | 0.8% |
| SnapMoGen `gpt` | 163,412 | 163,407 | **1.00** | — | 63 | 78.8% |
| SnapMoGen `manual` | 81,718 | 76,414 | 1.07 | — | 56 | 58.6% |
| Motion-X++ IDEA400 | 12,040 | 11,673 | 1.03 | 99.5% | 55 | 58.6% |
| Motion-X++ haa500 | 6,944 | 5,512 | 1.26 | 98.3% | 44 | 38.4% |
| Motion-X++ (7 subsets) | 25,579 | 23,436 | 1.09 | — | — | 52.0% |
| BONES-SEED `desc_1` | 142,220 | 5,272 | **27.0** | 0.27% | 15 | ~0% |
| AIOZ-GDANCE | 1,624 | **46** | **35.3** | — | ~8 | 0% |
| CoMPAS3D (one take) | 86 seg | 31 | 2.77 | — | ~12 | 0% |

**The decisive finding: caption uniqueness and token budget are anti-correlated.** Every dataset
reaching ≈1:1 does so by being *verbose*. SnapMoGen, IDEA400 and Motion-X perform are unique because
they are long, not because they are precise — and OMG's interface is `max_length=50`, hard-coded.

**MuRIL Devanagari fertility, measured on 6,000 real en/hi pairs: 1.25× median** (mean 1.24, p90
1.50) **[M]**. So English must be ≤40 t5 tokens for the Hindi to fit in 50 MuRIL tokens. Usable
counts under that gate:

| Dataset | **usable (Hindi-safe)** | survival |
|---|---:|---:|
| HumanML3D (+mirrors) | **85,209** | 97.5% |
| HumanML3D (originals) | 42,624 | 97.6% |
| SnapMoGen `manual` | 21,206 | 26.0% |
| SnapMoGen `gpt` | 9,831 | **6.0%** |
| Motion-X++ (7 subsets) | 7,857 | 30.7% |

SnapMoGen's most-unique split is **94% destroyed by the token gate**. Per-motion, even choosing the
*shortest* of its 6 captions leaves 41.9% of motions over 50 tokens.

**Ruled out structurally:** the dance family (AIOZ-GDANCE labels are literally
`id, music_genre, dance_style` — 46 labels for 1,624 sequences; CoMPAS3D is a finite salsa-move
vocabulary; FineDance/AIST++/ChoreoMaster/OpenDance are the same category). **Kungfu and BEAT2 carry
no text modality at all** per OMG's own Table 6 label column. **KIT-ML** uses a motion representation
incompatible with HumanML3D (BiMD states this), so it would need its own retargeting path for ~6,278
captions — poor value. **MotionGV, MotionLLaMA, FineDance, ChoreoMaster, OpenDance, PerMo are not on
HuggingFace at all** **[M]**.

**Windowing caveat that applies to every option:** OMG's own processed÷original expansion is a direct
caption-dilution proxy — LAFAN1 40→977 (24×), 100style 0.8K→10.7K (13×), FineDance 0.2K→6.0K (30×).
Every window inherits its parent's caption. **1:1 survives only if you take one window per motion.**

### 9.3 The choice

**A G1-retargeted AMASS/HumanML3D corpus already on local disk**, `g1_dataset_robot_v2.csv` — which
combines everything the search was looking for: near-1:1 captions, short captions that fit the frozen
interface, **already retargeted to G1** (no new conversion pipeline, no new silent-error surface),
and all four Indic translations already present.

---

## 10. The data as built

### 10.1 The source corpus

`D:\HumanML3d\g1_dataset_robot_v2.csv` — 13,282 clips, 27 columns **[M]**. Motion `.npz` files are
hosted at `AdiShingote/pragya-vla-g1-motion-dataset` on HuggingFace. Companion pipeline doc:
`D:\HumanML3d\HUMANML3D_TO_G1_PIPELINE.md` (ProtoMotions + PyRoki, 34-link / 29-DOF G1).

| property | value |
|---|---|
| Clips | 13,282 |
| `caption_1` fill | 100% · `caption_2` 99.93% · `caption_3` 98.77% · **`caption_4` 0.11%** |
| Languages | en + hi/bn/ta/te for each caption slot |
| Distinct `caption_1` | **12,508 → 1.062 clips/caption, 97.2% used once**, max reuse 24 |
| Source AMASS files | 10,480 (2,310 produce >1 clip, max 5) |
| Provenance | KIT 4,647 · CMU 2,913 · BMLmovi 1,839 · Eyes_Japan 1,465 · MPI_HDM05 771 · … |
| Duration | mean 7.37 s, median 7.95 s; **97.5% ≥ 2.33 s** (one L=10/H=60 window) |
| Mirrors | **none** |

Captions are rewritten to robot voice. Note a rewrite artifact: *"a Robot kicks something or
**Robot** with his left leg"* — "someone" was replaced by "Robot". Harmless for training; fix before
any caption appears in a paper figure.

**Three structural facts about the `.npz` files, verified 2026-09-05 [M] — the conversion depends
on all three:**

1. **`.npz` are per SOURCE MOTION, not per clip.** 10,480 files serve 13,282 rows. Download and
   decode each once, then cut every clip that references it.
2. **6,023 of 13,282 clips have `start_s > 0`.** A clip is the frame range `[start_s, end_s)`
   *inside* its source file. Ignoring this silently trains on the wrong motion for 45% of the corpus.
3. **Clips from one source overlap** — e.g. `ACCAD/.../A11 crawl forward` yields 0.40–10.40 s and
   7.05–17.05 s, sharing 3.35 s. This is independent confirmation that `source_amass` had to be a
   leakage axis in §10.3.

**Where the `.npz` actually live, and which token reaches them [M, 2026-09-05]:**

| repo | files | filename form | reachable by |
|---|---:|---|---|
| **`AdiShingote/pragya-vla-g1-motion-dataset`** | **13,741** | backslashes preserved — **matches the CSV `npz_path` verbatim** | the `AdiShingote` token only |
| `PragyaVLA/PragyaVLA-G1-motions-dataset` | 13,729 | non-alphanumerics collapsed to `_` | the `CodeSushh` token |
| `PragyaVLA/PragyaVLA-G1-125D-OMG` | 6,808 | HumanML3D indices (`000003.npz`) | `CodeSushh`; already 125-D, carries per-file `src_fps` |

The `AdiShingote` repo is the original: its 12 extra files are exactly the ones the CSV references
but the PragyaVLA copy lacks, and CSV URLs resolve **verbatim** with that token (12/12 probed). With
the `CodeSushh` token the PragyaVLA copy resolves only via a normalization
(`re.sub(r"[^A-Za-z0-9._]", "_", filename)`), reaching 13,269/13,282 = 99.90%.

**The source frame rate is PER-SUBSET, not global [M]** — measured exactly on single-clip sources
with `start_s == 0`, where the clip is the whole file, so `fps = T / duration_s`:

| subset | fps | | subset | fps |
|---|---:|---|---|---:|
| ACCAD | 30 | | EKUT | 25 |
| BMLmovi | 30 | | KIT | 25 |
| CMU | **15** | | SFU | **15** |

Three distinct rates (rounded histogram `{15: 14, 25: 26, 30: 13}`). **A single `--src-fps` is
therefore the wrong design** and would silently resample most of the corpus at the wrong rate — the
exact failure class §4.2 was written to prevent. The converter must carry a per-subset table.

**Truncation [?]** — several sources returned exactly `T = 225` frames, and under the correct
per-subset rates **26.7% of sampled clips reference frames past the end of their file** (worst case:
a CMU clip claiming `end_s = 28.05` against a 225-frame file holding ~15 s). Two separate causes:
off-by-one rounding (harmless, clamp it) and genuine truncation (data loss). Note `duration_s` is
capped at 10.00 s for 36.5% of clips, so segmentation and export used disagreeing caps. Local
`amass_test/retargeted-g1/` files show **no 225 cap** (max T = 261), so the cap may be specific to
the uploaded copies. Resolution pending; do not convert until settled.

### 10.2 Token budget — the constraint that killed the alternatives, cleared here

Measured on the real translations with the real tokenizers **[M]**:

| field | n | median | mean | p95 | **>50** |
|---|---:|---:|---:|---:|---:|
| caption_1/2/3 (t5, en) | 13,282 / 13,273 / 13,119 | 15 | 17.1 | 34–35 | 0.68–0.75% |
| **Hindi (MuRIL)** | 39,674 | 17 | 19.9 | 41 | **1.90%** |
| **Bengali (MuRIL)** | 39,674 | 14 | 16.6 | 33 | **0.52%** |
| **Tamil (MuRIL)** | 39,674 | 16 | 18.4 | 35 | **0.80%** |
| **Telugu (MuRIL)** | 39,674 | 19 | 21.7 | 42 | **2.09%** |

**All four languages clear the 50-token interface with ≥97.9% coverage. No resampler is needed** —
the interface validated in Run 1 stays untouched, which is precisely what keeps Run 2's results
attributable to the objective rather than to a new module.

Total: **39,674 (clip, English caption) pairs → 158,696 (clip, Indic caption) pairs.**

### 10.3 The splits

Built by `OMG/tools/build_amass_g1_splits.py` → `OMG/splits_amass_g1_13k/` **[M]**.

**Two leakage axes**, both real in this corpus and both closed: (1) `source_amass` — one AMASS take
segmented into up to 5 clips, 2,310 sources affected; (2) `caption_1` — 774 clips share a caption, and
because the Indic captions are *derived* from the English, a shared English caption means a shared
Hindi/Bengali/Tamil/Telugu caption. Clips are merged transitively (union-find) by shared source **or**
shared caption; splits are drawn over **groups**, never clips. Allocation is stratified per AMASS
subset. 331 clips shorter than 2.33 s are dropped.

12,951 usable clips → **9,417 leakage-safe super-groups**.

| split | clips | groups | captions | clips/cap | EN pairs | Indic pairs | hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| **train** | 9,941 | 7,265 | 9,396 | 1.058 | 29,712 | **118,848** | 20.7 |
| **val** | 1,007 | 719 | 936 | 1.076 | 3,011 | 12,044 | 2.1 |
| **test** | 1,010 | 719 | 950 | 1.063 | 3,020 | 12,080 | 2.1 |
| **reserve** | 993 | 714 | 934 | 1.063 | 2,963 | 11,852 | 2.1 |

**Independently re-verified after writing [M]:** zero shared captions and zero shared source files
across all six split pairs. Subset balance holds tight (KIT 35.6/35.4/35.4/35.1%, CMU
21.2/21.3/21.5/20.7%).

**Training budget:** 1 window/clip = 29,712 samples per language (strict 1:1); **2 windows/clip
(Run-1 precedent, stride 30) = 59,424 per language, 237,696 across four.** Recommended: 2 windows —
median clip is 7.95 s, so two 70-frame windows are genuinely different motion, not a crop.

`reserve` is untouched and exists solely for native-speaker authoring (§3.3, §8.2).

### 10.4 The mirror gap

**The retargeted corpus contains no mirrors** — HumanML3D's M-prefixed files were not carried through
retargeting **[M]**. This matters because HumanML3D's mirrors ship **human-written left/right-swapped
captions**: 8,953 of 14,616 pairs (61.3%) have genuinely different caption sets, 8,413 because the
original mentions left or right **[M]**. That would have been Contribution 2's entire supervision
signal, free.

It must now be manufactured: mirror the G1 motion programmatically (swap L/R joint pairs, negate the
lateral axis and the corresponding rotation components) and swap caption direction words using the
inventory already built in `direction_minimal_pairs.py` (right = दाहिनी, दाहिने, दाईं, दाएं, दाहिना,
दायीं, दाएँ; left = बायीं, बाएं, बाईं, बाएँ, बायां, बायाँ, with बा- false positives explicitly
excluded). **The pool is healthy: 34.4% of training captions carry a direction word** (10,225 English,
10,169 Hindi) **[M]**.

---

## 11. Run 2 experiment plan

**11.0 — Gate: measure `L_null` first.** Score the cached t5 `""` null context under the fixed-seed
`finalize_run.py` protocol. Ten minutes. It converts §7.2's collapse economics from estimate to
measurement and sets `λ_x`. **Do not tune anything before this.**

**11.1 — Symmetric ablation, one language.** Keep everything from Run 1 but replace the English
teacher with a *second caption of the same clip* (`caption_1_hi` vs `caption_3_hi`) — symmetric,
gradient on both sides, no English target. One ~4 h run, scored on the same fixed-seed protocol
against the step-22000 numbers. **This isolates the objective change from the multilingual change.**
If it collapses, that is learned for the price of one afternoon rather than a whole matrix.

**11.2 — Two languages, sanity.** Hindi + Tamil, `λ_x = 0.5`, `λ_en = 0.1`. Watch per-language
caption-swap sensitivity and text-induced variance every validation.

**11.3 — All four, the method run.** Shared adapter, sampled language pairs, non-parallel caption
pairing (different caption slot *and* different language, e.g. `caption_1_hi` ↔ `caption_3_ta`),
small English anchor. Baselines: four per-language adapters, and shared-but-unaligned (native loss
only). **This is the method table.**

**11.4 — Held-out language.** Within-family (train hi+bn+ta, test **te**) and cross-family (train
hi+bn Indo-Aryan, test **ta**). Then the few-shot curve at 0 / 1 / 10 / 100% of the held-out
language's pairs — the language-axis analogue of OMG's own Table 4. **This is the figure.**

**11.5 — Contribution 2, concurrent with 11.3.** Manufacture mirrors per §10.4; train counterfactual
grounding into the adapter; escalate to rank-8 K/V LoRA only if the frozen prior cannot express it.
Evaluate at n ≥ 550/arm.

**Cost [E]:** ~4.5 h per twin-branch run at 1.5–1.8× Run 1's 0.393 s/step; the full matrix
(4 per-language baselines + shared + method + 2 ablations + 2 holdouts + 6 short few-shot fine-tunes)
lands near **45 GPU-hours ≈ $20–30** on a rented 4090. **Compute is not the constraint; annotation
is.**

---

## 12. Metrics and diagnostics

| metric | what it answers | notes |
|---|---|---|
| `L_native` | is the motion right | **weak signal for text grounding** — ~95% of achievable reduction saturates early |
| `rel` (caption swap) | does the caption matter at all | RANDOM −1.17% / step-22000 +22.54% / ENGLISH +28.68%. **Anchor-relative — never port a "% of gap" figure across runs** |
| **text-induced variance** | early-warning for partial collapse | new; per language; **never optimized** |
| **CLRD** (cross-lingual response divergence) | do all languages do the same thing | `E_motion mean_{ℓ≠ℓ'} ‖x̂₀(ℓ) − x̂₀(ℓ')‖` — needs no evaluator, no English, no gallery. Report beside a collapse check: CLRD≈0 *and* sensitivity≈0 means the model ignores text, not that it agrees |
| R@1 / FID / KID / Diversity | comparability to Run 1 and to OMG | via OMG's own metric functions, applied to our test split |
| direction flip rate | attribute grounding | English calibration arm mandatory; n ≥ 550 for significance |
| tracker metrics | physical executability | contact slide, jerk, MPJPE, falls, joint-limit violations |

---

## 13. Open questions and risks

1. **`L_null` unmeasured [?]** — §11.0. Gates everything in §7.2.
2. **Mirrors must be manufactured** — §10.4. Contribution 2 depends on it.
3. **Every caption is still machine-translated.** The `reserve` split exists for native authoring and
   remains untouched. This is the single largest gap between the work and a claim that survives review.
4. **Zero-shot transfer across the Indo-Aryan → Dravidian boundary is the biggest empirical unknown.**
   Medium risk; the few-shot curve makes any outcome publishable.
5. **The frozen prior may lack directional capacity even with K/V LoRA.** A clean negative is still a
   finding, but it weakens the headline.
6. **Hindi retrieval is English-mediated** (§2.3). The MuRIL-based evaluator (§8.2) is the fix.
7. **Register heterogeneity** if datasets are ever mixed — Run 1's §6.2 audit identified register
   drift as likely the dominant FID driver for MT_PIVOT. A reason to stay on one corpus.
8. **Annotation is the schedule risk, not compute.** Recruit verifiers early.

---

## 14. Errors and lessons

Continuing the numbering from `MAILA_RESEARCH_LOG_2026-08-22.md` §11 (which holds errors 1–8).

9. **The RANDOM anchor was scored with adapter dropout ON while every trained checkpoint was scored
   with it OFF** **[C]**. `build()` ends with `model.eval(); model.text_encoder.adapter.train()`,
   leaving `Dropout(0.1)` active; the RANDOM row calls `build()` and scores immediately, while every
   trained row calls `m.text_encoder.adapter.eval()` first. (ENGLISH_t5 is unaffected —
   `passthrough_t5` bypasses the adapter.) Magnitude is probably small — dropout hits only the
   residual branch, scaled by α=0.1 — but the anchor defining the zero point of `pct_of_gap` was
   measured under a different configuration from the rows it anchors. A second, independent reason
   never to port `pct_of_gap` across runs.

10. **The Run-1 log's §3.2 describes an architecture the code does not implement** **[C]**. It
    describes *50 learned queries* and a *predicted-length mask* (`mask_from_length_logits`). The
    shipped `maila_encoder.py` has neither: it is a per-token residual MLP over MuRIL states, and the
    mask comes straight from MuRIL's own attention mask, with the fixed `[B,50,768]` contract
    satisfied by `padding="max_length"` truncation. **The contract is met; the mechanism described is
    not the one that ran.** Must be corrected before any paper draft.

11. **Machine translation silently collapsed distinct English captions in the training corpus**
    **[M]**. 12 English captions across 149 BONES-SEED train clips (0.50%) map to identical Hindi,
    systematically on gait: *walk* and *jog* become the same Hindi verb. Total collapses are only the
    cases where two strings became byte-identical; partial degradation of speed and manner is
    invisible to this test and likely more widespread. Same failure class as the
    `pirouettes → समुद्री डाकू` case in Run 1 §6.2, but on the input side.

12. **Motion-X / IDEA400 ship LLM refusal strings as captions** **[M]** — *"sorry, i can't assist
    with that request"* appears 225× in idea400, ~300 across the seven subsets. If any Motion-X-derived
    source is ever used, filter these.

13. **A near-1:1 caption ratio can be an artifact of verbosity rather than precision** **[M]**.
    SnapMoGen `gpt` scores 1.00 clips/caption and is 94% unusable at a 50-token interface. Uniqueness
    must always be read together with the token budget of the consuming model.

14. **THE ADAPTER COULD NEVER REACH t5 GEOMETRY — `nn.LayerNorm` was the last op** **[M]**
    *(2026-09-06; the single most consequential error in the project so far)*

    `ResidualAdapter.forward` ends `return self.out_norm(u + self.alpha * r)`. `out_norm` is
    `nn.LayerNorm(768)` with PyTorch's default affine (weight=1, bias=0), which normalises each
    token to zero mean and unit variance. Unit variance across 768 channels means the output
    token norm is **exactly sqrt(768) = 27.71** — for every caption, at every step, under every
    value of `alpha`, and no matter what the MLP learns.

    t5-base's real contexts have token norm **6.758**. OMG's `proj` is `nn.Identity()`, so
    nothing sits between the encoder and cross-attention to absorb the difference. The frozen
    DiT was reading context vectors **4.1x too long** in Run 1 and in every Run 2 attempt.

    Measured directly, 1,024 val windows, replayed seeds, adapter otherwise untouched:

    | adapter state | L_correct | L_wrong | rel | tok_norm | cos(per-channel mean, t5) |
    |---|---|---|---|---|---|
    | random init, `out_norm` w=1 b=0 (as built) | 0.07454 | 0.07338 | −1.56% | 27.713 | 0.043 |
    | random init, `out_norm` recoloured to t5   | 0.03883 | 0.03884 | +0.04% |  7.396 | — |
    | trained lambda_en=0.1 @ step 8,000 (as built) | 0.02574 | 0.02603 | +1.10% | 27.143 | 0.133 |
    | trained lambda_en=0.1 @ step 8,000 + recolour | 0.02716 | 0.02724 | +0.30% |  7.491 | — |
    | t5-base(english), the DiT's own reference | — | — | +13.75% | 6.758 | 1.000 |

    **Roughly half the initial denoising loss was scale error, not semantics.** That is what the
    failed run was minimising: across 8,000 steps L fell 0.0745 → 0.0257 while `cos` crawled
    0.043 → 0.133 and `rel` never left zero. The optimiser was grinding 768 LayerNorm gains down
    toward 0.25 instead of learning Hindi → motion.

    **This invalidates the previous diagnosis recorded in §14 item 11 and in the `--lambda-en`
    help text.** The lambda_en=0.1 failure was blamed on a weak English anchor. It was not a
    weight problem: no value of lambda_en can move a quantity that LayerNorm pins. Raising
    lambda_en 0.1 → 0.5 would have partly masked the symptom by supplying a stronger gradient
    to the same uphill fight, which is likely how Run 1 reached +11.92% *despite* the bug.

    **Fix — recolour, not repair.** LayerNorm already emits zero-mean/unit-variance `z` and then
    applies weight/bias, so initialising `weight[d] = std_t5[d]`, `bias[d] = mean_t5[d]` moves the
    starting point into t5's marginal distribution at **zero parameter and zero capacity cost**.
    Measured on 34,258 valid tokens from 2,048 **train-split** English captions
    (`tools/calibrate_t5_geometry.py`, sha256 `8295abddff521ef2…`): mean |mu| 0.0691,
    mean sd 0.2307, implied token norm 6.977.

    It must be an **initialisation**. Recolouring an already-trained checkpoint made it *worse*
    (rel +1.10% → +0.30%) because those weights had adapted to the 27.7 scale. `maila_train.py`
    now refuses to start if the init token norm falls outside (4, 12).

    **Lesson — the project's recurring failure class, in its purest form yet.** "Shapes match,
    distributions don't." `(B, 50, 768)` was correct at every step; `state_dict` loaded cleanly;
    no assertion fired. The bug was visible only by measuring the *statistics* of a tensor whose
    *shape* was right. `maila_encoder.py`'s own docstring states the invariant it violated:
    *"the adapter output must land in t5-base's ACTUAL geometry."* Writing the invariant down is
    not the same as testing it. **Every frozen interface needs a distributional gate, not just a
    shape check** — and the gate belongs in code that runs, not in a docstring.

    **Cost of not measuring earlier:** ~14 GPU-hours across two runs, plus Run 1's headline
    number being an underestimate of what the architecture can do.

15. **THE ADAPTER EMITTED A NEAR-CONSTANT VECTOR — `nn.LayerNorm` CANNOT REMOVE A SHARED
    COMPONENT** **[M]** *(2026-09-06; found immediately after #14, and larger than it)*

    Fixing #14 dropped the token norm to t5's scale but `rel` stayed at ~0 and `tiv` at
    2.4e-04. Measuring *how much the context moves when the caption changes* explained why.

    Decomposing each encoder's output into the component every caption shares (`mu`) and the
    caption-specific deviation, over 512 val captions:

    | encoder | ‖mu‖ | between-caption | btwn/mu | cos(i,j) |
    |---|---|---|---|---|
    | t5-base(english), the DiT's reference | 2.644 | 1.904 | **0.7201** | **0.6563** |
    | adapter(hindi), #14 recolour only | 7.285 | 0.423 | **0.0581** | **0.9966** |

    `cos(i,j)` is the mean cosine between the contexts of two **different** captions. At
    **0.9966** the adapter was handing the frozen cross-attention very nearly the same vector
    for every caption. ~7.3 units of constant, ~0.4 units of content — **8.1% of t5's
    caption-specific signal.** No value of any loss weight can condition on that.

    **Cause.** MuRIL is anisotropic, and `nn.LayerNorm` standardises *within* a token, across
    the 768 channels. A direction shared by *all* tokens is invisible to it — LayerNorm
    rescales that direction along with the signal but never removes it. Measured
    ‖mu_corpus‖ = **26.87** against a token norm of 27.71: the shared component *was*
    essentially the entire vector. #14's recolour faithfully rescaled it too.

    **Fix — normalise with corpus statistics, not within-token statistics.**
    `z = (y − src_mu) / src_sd`, then `z * sd_t5 + mu_t5`, where `src_mu`/`src_sd` are MuRIL
    per-channel statistics measured across the TRAIN split (`tools/calibrate_adapter_geometry.py`,
    sha256 `ca74f04c77fd4375…`). Measured at init, no training:

    | init | btwn/mu | cos(i,j) | L_correct |
    |---|---|---|---|
    | LayerNorm + t5 recolour (#14 fix alone) | 0.0581 | 0.9966 | 0.03005 |
    | **centre + per-channel std + recolour** | **0.8042** | **0.6080** | **0.02778** |
    | ZCA whiten + recolour | 0.8925 | 0.5628 | 0.02807 |
    | t5-base(english) | 0.7201 | 0.6563 | — |

    **13.8x more caption signal**, and closest to t5. ZCA was rejected: after centring, the top
    eigenvalue holds only 9.9% of the variance — *the mean was the anisotropy* — so full
    decorrelation overshoots t5's own structure for no measured gain.

    A side effect that matters: LayerNorm-last also **pinned the output scale**, which is why
    the residual branch could never gain authority over the identity path (Run 1: alpha moved
    0.100 → 0.131 across 25,000 steps, just 1.31x). Corpus normalisation leaves the scale free
    to grow with `alpha`.

    **Live confirmation.** On relaunch the consistency term `xling` rose from 0.00014 to
    0.00089 at comparable steps — ~6x more gradient, because the two views finally differ.

    **Lesson — and the standing fix.** #14 and #15 are the same failure at two moments: a
    tensor whose *shape* is right and whose *distribution* is wrong. `(B, 50, 768)` was correct
    throughout; `state_dict` loaded cleanly; nothing asserted. `maila_train_motionpivot.py` now
    runs `geometry_gate()` before training and **refuses to start** unless the init token norm
    is in (4, 12) and `cos(i,j) <= 0.95`. **Every frozen interface gets a distributional gate
    that runs — a docstring stating the invariant is not a test of it.**

    **Correction to the record:** the `lambda_en` diagnosis in item 11 and the #14 entry were
    both incomplete. Arms at lambda_en 0.1 and 0.5 were indistinguishable (rel −0.49% vs
    −0.48% at step 1,000) because neither could matter while `cos(i,j) = 0.9966`.

16. **"ENGLISH_t5" IS NOT A CEILING, AND EARLIER pct_of_gap NUMBERS WERE INVALID** **[M]**
    *(2026-09-06, from the Run 2 final evaluation)*

    Measured on TEST, 1,024 windows, per-batch seeds, identical protocol for every row:

    | reference | L_correct | rel | rel_draw |
    |---|---|---|---|
    | NULL, text-blind | 0.04088 | +0.00% | +0.00% |
    | untrained adapter, corpus init | 0.03473 | +2.13% | +1.82% |
    | English t5, frozen, zero-shot on this corpus | 0.03749 | +9.80% | +8.83% |
    | arm C, Hindi, lambda_en 0.1, step 20k | 0.03396 | +10.19% | +7.97% |
    | arm D, Hindi, lambda_en 0.5, step 20k | 0.03414 | +10.46% | +8.23% |

    **Both trained Hindi arms beat frozen English t5, on caption sensitivity AND on absolute
    loss.** So the English arm is not an upper bound and never was: it is a ZERO-SHOT
    baseline. The adapter has 25,000 steps of adaptation to this corpus; t5 has none. Any
    trained conditioner can pass it.

    Consequences, stated plainly:
      * every `pct_of_gap` figure computed against an "English ceiling" -- including the
        "55.9% of gap vs Run 1's 51.9%" reported mid-run -- is **withdrawn**. The denominator
        was not a ceiling, and the +13.75% version of it was additionally measured under the
        4-seed bug (#17).
      * this is NOT evidence that Hindi conditioning beats English conditioning. The honest
        claim is narrower: *a corpus-adapted Hindi adapter beats zero-shot frozen English t5
        on this corpus.* The missing comparison is an ENGLISH adapter trained identically on
        the same corpus. Until that ablation exists, no cross-language superiority claim can
        be made, and none should appear in the paper.

    **The NULL row is the harness self-check.** With the cached t5 null context every caption
    yields the same context, so `rel` must be exactly 0 and `tiv` exactly 0. Both are, to
    printed precision -- the measurement is not manufacturing sensitivity.

17. **`rel` DEPENDS ON THE SEEDING AND ON WHICH WRONG CAPTION YOU PICK** **[M]** *(2026-09-06)*

    `eval_run2.py` first cycled four fixed seeds across all 32 validation batches, giving 128
    distinct (tau, epsilon) draws instead of 1024. Those four landed on low-noise timesteps.
    Same checkpoint, same split:

    | protocol | L_correct | rel |
    |---|---|---|
    | trainer: one seed per batch (correct) | 0.03317 | **+7.42%** |
    | same, fp32 instead of bf16 autocast | 0.03319 | +7.35% |
    | 4 cycling seeds (the bug) | 0.02483 | +2.59% |
    | one seed per batch, wrong caption DRAWN from split | 0.03319 | **+11.13%** |

    Precision is irrelevant (0.07 pp). The seed count moved `rel` by 4.8 pp. Caught only by
    refusing to accept a mismatch against the training log.

    Second axis: rolling the wrong caption WITHIN a batch understates sensitivity, because a
    clip sometimes lands beside a near-duplicate caption and MT collapse maps distinct English
    onto identical Indic strings. Drawing from the whole split gives +11.13% vs +7.42%.
    `eval_run2.py` now reports both (`rel` for comparability with Run 1, `rel_draw`).

18. **THE PHYSICAL METRICS DID NOT DISCRIMINATE, AND CFG SELECTION BY MPJPE IS BIASED** **[M]**

    On TEST, 3 seeds, both arms: joint-limit rate, hard-limit rate, joint-jump rate, ground
    penetration rate and depth, and fall rate are **all exactly 0.000 +/- 0.000**. On 2-second
    (60-frame) windows anchored at the canonical root, these thresholds are simply never
    approached. They are evidence of *no gross failure*, not evidence of quality, and they
    separate nothing. Only foot sliding varies meaningfully (arm C 0.0344, arm D 0.0334 m/s;
    slide rate 4.7% vs 4.3%) along with jerk (3.68 vs 3.69) and MPJPE (36.9 vs 36.4 mm).

    Also: MPJPE rises monotonically with guidance (2.0 -> 35.9, 2.5 -> 39.6, 3.0 -> 43.6 mm),
    so **selecting CFG by MPJPE always picks the lowest CFG on offer**. Guidance trades
    fidelity for text adherence, so a fidelity-only criterion is the wrong selector. CFG
    should be chosen on a text-adherence metric (R-precision via three_arm_benchmark.py), and
    the reported cfg=2.0 should be treated as provisional.

19. **RUN 2 FINAL: THE ENGLISH ABLATION, AND THE BASELINE NOBODY HAD MEASURED** **[M]**
    *(2026-09-06; four arms, 25,000 steps each, TEST split, one protocol for every row)*

    | configuration | rel | rel_draw | L_correct | MPJPE |
    |---|---|---|---|---|
    | null, text-blind floor | +0.00% | +0.00% | 0.04088 | -- |
    | **stock OMG given HINDI** | **+1.80%** | +0.46% | 0.03914 | -- |
    | untrained adapter, hi corpus init | +2.13% | +1.82% | 0.03473 | -- |
    | untrained adapter, en corpus init | -1.55% | +0.17% | -- | -- |
    | stock OMG given ENGLISH, zero-shot | +9.80% | +8.83% | 0.03749 | -- |
    | MAILA hindi, arm C (lambda_en 0.1) | +10.19% | +7.97% | 0.03396 | 36.9 mm |
    | **MAILA hindi, arm D (lambda_en 0.5)** | **+10.46%** | +8.23% | 0.03414 | 36.4 mm |
    | MAILA english, arm E (lambda_en 0.1) | +13.44% | +10.33% | 0.03476 | 34.8 mm |
    | **MAILA english, arm F (lambda_en 0.5)** | **+14.69%** | +10.79% | 0.03435 | 34.6 mm |

    Hindi arms selected step 20,000; English arms step 18,000; cfg 2.0 throughout; 0 eval errors.

    **THE BASELINE THAT MATTERS, and that had never been scored until the last hour of the
    run: stock OMG handed a Hindi caption reaches +1.80%, and +0.46% on rel_draw** -- against
    a text-blind floor of exactly 0.00%. t5-base cannot read Devanagari: two different Hindi
    captions leave it at cosine 0.9966 apart (SSOT #15 measured the same collapse). The
    adapter lifts that to +10.46%, a **5.8x gain on rel and 17x on rel_draw**, and it exceeds
    zero-shot English t5 (+9.80%) while reading Hindi.

    **But Hindi reaches only 71% of an identically-trained English adapter** (10.46 / 14.69;
    76% on rel_draw), and English also generates better motion (MPJPE 34.6 vs 36.4 mm, foot
    sliding 0.0225 vs 0.0334 m/s). **Any claim that Hindi matches English is false.** The
    mid-run figure of "104% of English capability" measured against ZERO-SHOT t5 and is
    withdrawn -- see #16. Candidate causes for the residual gap, both already measured:
    MuRIL needs 1.20-1.25x more tokens for the same content in Devanagari against a hard
    50-token window, and the Hindi captions are machine-translated, which collapses distinct
    English strings into identical Indic ones (#15, and the split-leakage finding).

    **lambda_en FLIPS SIGN BY LANGUAGE, and this is evidence for the pivot.** English prefers
    the strong anchor (+14.69 vs +13.44); Hindi prefers the weak one (+10.46 vs +10.19, and
    +7.89 vs +7.43 on val). For an English arm the anchor distils t5 onto the SAME string, so
    it is free supervision; for a Hindi arm it is a cross-lingual pull competing with the
    motion pivot. The pivot is doing work rather than riding on English distillation.

20. **A TEARDOWN GATE MUST NOT VERIFY THE LOG IT IS WRITING** **[M]** *(2026-09-06)*

    `finish_and_destroy.sh` uploaded every artifact, verified 106/107, and refused to destroy
    the instance over one mismatch: `logs/12_finish.log` -- its own log. The chain uploads it,
    then writes `gate exit=...` into it, so the local copy diverges the instant the result is
    recorded. **That file can never verify while the chain that writes it is running.**

    The gate behaved correctly (it refused rather than guessed, and the box survived), but the
    failure was structural, not a data problem. Re-running the gate after the chain exited gave
    107/107. Fix for any future chain: exclude the chain's own live log from the verified set,
    or write the gate result to a separate file the gate does not cover.

    Related, same session: a backgrounded `scp -r` of the run directories exited 0 having
    copied **1 of 26** English checkpoints. Caught only by counting files afterwards rather
    than trusting the exit code. **Verify by counting the artifacts, never by the exit status
    of the command that was supposed to produce them.**

21. **eval_run2.py DOUBLE-DENORMALISED EVERY GENERATED MOTION** **[M]** *(2026-09-06)*

    `motion_generator.generate()` returns **denormalised** features
    (`features = self.representation.denormalize_features(future_norm)`), but `rep.decode()`
    calls `denormalize_features` again. `eval_run2.physical_and_fidelity` fed
    `g["motion_features"]` straight into `decode`, so every generated motion was scaled wrong.
    Measured on test clip 98:

    | path | root-XY travel |
    |---|---|
    | `generate()["qpos_36"]` (authoritative) | 1.733 m |
    | `decode(generate feats)` -- what the eval did | 0.409 m |
    | `decode(normalize(generate feats))` -- the fix | **1.733 m** |

    Re-normalising first reproduces `qpos_36` exactly, which both confirms the diagnosis and
    proves the inverse is correct. Motion was compressed to ~24% of true scale.

    **This is why the physical metrics were all exactly 0.000 +/- 0.000** (#18). Motion at a
    quarter scale never approaches a joint limit, never penetrates the ground, never falls. At
    the time this was written up as "the metrics do not discriminate". The truer reading was
    "this metric is not measuring what I think it is" -- a degenerate metric is a bug report,
    not a finding.

    CORRECTED (test, cfg 2.0, 3 seeds). rel / rel_draw / tiv / L_correct are UNAFFECTED --
    they come from `native_loss` on the model and never touch the decode path:

    | metric | hindi C | hindi D | english E | english F |
    |---|---|---|---|---|
    | MPJPE (mm) | 113.5 | 111.6 | 113.4 | 111.6 |
    | global MPJPE (mm) | 280.2 | 275.1 | 258.1 | **255.5** |
    | e_vel | 10.57 | 10.47 | 9.65 | **9.54** |
    | foot sliding (m/s) | 0.159 | 0.140 | 0.134 | **0.133** |
    | body jerk | 12.25 | 12.15 | **11.73** | 11.96 |
    | fall rate | 0.017 | 0.017 | 0.021 | 0.018 |
    | joint-limit rate | 0.172 | **0.156** | 0.224 | 0.241 |

    **Local pose fidelity is TIED** (MPJPE 111.6 for both best arms). The earlier claim that
    English generates better motion (34.6 vs 36.4 mm) was entirely the scale bug. The real
    difference is GLOBAL TRAJECTORY: English 255.5 vs Hindi 275.1 mm, velocity error 9.54 vs
    10.47 -- the Hindi arm gets the pose as right and drifts more on where the body goes,
    consistent with Run 1's direction-word failure. Hindi violates joint limits LESS
    (0.156 vs 0.241); unexplained.

22. **[WITHDRAWN -- see #24] CAPTIONS DESCRIBE THE SOURCE CLIP; WINDOWS ARE SLICES** **[M]**
    *(2026-09-06, found only by rendering)*

    Every generated arm travels far further than ground truth, INCLUDING STOCK OMG ON ENGLISH
    (clip 98: gt 0.20 m, omg_en 1.73 m, omg_hi 1.39 m, maila_hi 2.39 m). So it is not a Hindi
    effect and not an adapter effect.

    The GT decode is NOT losing translation. If it were, feet would swing through a full gait
    (~0.5-0.8 m relative to the pelvis) while the pelvis stayed put. Measured over 125 test
    clips: foot swing relative to pelvis median **0.034 m**, root displacement median
    **0.019 m**, max joint range median 0.41 rad. Both tiny -- the windows are genuinely
    near-static.

    The cause is the pairing. Clip 98's caption is *"the Robot walks forward, then walks to
    their right and steps up and down as if stepping over or on obstacles"* -- a multi-second
    sequence -- while its 60-frame window holds a 20 cm shuffle. **The caption describes the
    whole source clip; the training window is a slice of it.** Text and motion are only
    loosely paired by construction.

    This is the best available explanation for why this corpus's English ceiling is ~+9.8%
    where Run 1's bones-seed corpus reached +28.68% (#16), and it reframes that ceiling as a
    DATA-CONSTRUCTION limit rather than a method limit. It also means a model generating 1.7 m
    of walking may be following the caption BETTER than the ground-truth fragment does, so the
    gt column in the rendered panels is "what this window contains", not "what the caption
    asked for".

    **Implication for the next experiment:** the highest-value change is not architectural. It
    is re-windowing so that each caption describes the segment it is paired with -- or
    scoring only on windows whose caption plausibly covers them.

23. **A VERIFICATION GATE IS ONLY AS GOOD AS ITS MANIFEST** **[M]** *(2026-09-06)*

    `adapter_calibration_en.pt` was never pushed to the Hub: `sync_artifacts.targets()`
    hard-codes a provenance filename list written before the English ablation existed, so the
    English calibration fell outside the manifest. `teardown_gate.py` still passed 107/107 --
    it verifies that every file the manifest NAMES is intact, and cannot know a file is
    missing from the list. The destroyed box held the only copy.

    Recovered because the calibration is deterministic from the train split at seed 0:
    regenerating gave sha256 `3ae6521b69866936`, identical to the value recorded before
    teardown, so the English arm is provably the model that scored +14.69%.

    Fix for any future run: derive the provenance set by globbing what the run actually
    produced, never from a hard-coded list.

24. **THE SHARDS STORE RAW FEATURES; EVERY DECODE OF THEM WAS DOUBLE-DENORMALISED** **[M]**
    *(2026-09-07, found because the user said the ground-truth panel looked stuck)*

    `convert_g1_npz_to_omg125.encode_windows()` ends with `rep.codec.assemble_features(comps)`
    -- **raw**, not normalised. It never calls `rep.encode()`. But `rep.decode()` begins with
    `denormalize_features()`, so decoding a shard directly applies a spurious SECOND
    denormalisation and compresses motion to a near-frozen pose.

    The converter's own `verify_roundtrip()` shows the intended inverse: it uses
    `codec.split_features()` then `codec.decode_to_world_qpos36()`, bypassing the
    normalise layer. `rep.decode(rep.normalize_features(x))` is the equivalent, and it
    reproduces `generate()["qpos_36"]` exactly (1.733 m on clip 98, three decimals).

    **TRAINING IS UNAFFECTED.** Verified against upstream
    `motion_generator.py:486`: `_target_sequence` calls `self.representation.encode(batch)`,
    which normalises internally, and `_history_features` is passed through
    `normalize_features` at :491. Raw shards are the CORRECT contract -- the model normalises
    for itself, and `generate()` returns raw features symmetrically. So every conditioning
    result stands: rel, rel_draw, tiv, L_correct, the +1.80% stock-OMG-on-Hindi baseline, the
    +10.46% / +14.69% arms, and the 71% ratio.

    **WHAT IS INVALID.** #21 fixed only the PREDICTION side and left the REFERENCE side
    broken, so it compared a correct sample against a corrupted target:
      * MPJPE, global MPJPE, e_vel, e_acc -- invalid in BOTH the original and the #21
        "corrected" numbers. 36.4 mm and 111.6 mm are both wrong.
      * the rendered `gt` column in all 20 panels -- visibly frozen, which is how it was caught.
      * generated-only metrics (foot sliding, jerk, joint-limit rate, fall rate, ground
        penetration) are VALID after #21, since they never touch the reference.

    **ITEM 22 IS WITHDRAWN IN FULL.** The claim that windows are "genuinely near-static" was
    measured through this same broken decode -- foot swing 0.034 m, root 0.019 m were both
    artefacts. Everything built on it collapses with it: the caption-window misalignment
    story, the reading of the ~+9.8% English ceiling as a data-construction limit, and the
    recommendation to re-window the corpus. None of it is supported. The ceiling explanation
    of #16 (this corpus simply carries less text signal than bones-seed) stands on its own
    separate evidence and is not affected.

    **LESSON, and it is the third instance of the same one.** #14 and #15 were shape-correct /
    distribution-wrong. #21 and #24 are the same failure at the codec boundary: a tensor of the
    right shape on the wrong side of a normalisation. Two conventions meet here -- shards and
    `generate()` speak RAW; the diffusion model speaks NORMALISED -- and nothing in a shape
    check can tell them apart. **Any code crossing that boundary must state which side it is
    on.** A decode helper that asserts its input scale against `rep.mean`/`rep.std` would have
    caught all of it.

    **AND: I explained the symptom away instead of chasing it.** The frozen ground truth was
    visible in the very first smoke test (gt 0.20 m vs generated 2.39 m). I wrote a diagnostic,
    ran it through the same broken decode, and concluded the data was static -- turning a bug
    into a dataset finding. A measurement that "explains" an anomaly using the same code path
    that produced it proves nothing.

---

## 17. Run 3 plan — scaling to four Indic languages *(2026-09-07, all figures [M])*

### 17.1 The corpus source: OMG-Data, natively G1

`THU-MARS/OMG-Data` (public, `private=False`, 62.15 GB, LeRobotDataset-v3.0) is OMG's entire
1,174-hour pretraining corpus published **already in Unitree G1 `qpos_36` at 30 fps** --
exactly the input `convert_g1_npz_to_omg125.py` takes. `observation.state[36]` is
root_pos(3) + root_quat(4) + joints(29). 798,181 episodes / 121,171,939 frames /
590,722 tasks; 771,615 episodes (96.7%) carry text.

**This eliminates SMPL->G1 retargeting entirely, and eliminates the `SUBSET_FPS` table** --
the per-subset fps measurement that produced a wrong global `--src-fps` flag, a wrongly
retracted fps table, and a basename-collision bug. OMG-Data is uniformly 30 fps.

`meta/episodes/*.parquet` carries `omg/dataset`, `omg/source_id`, `omg/split`,
`omg/segment_index`, and per-modality flags, so it is fully filterable.

**Corrects SSOT s9.2**, which recorded that "MotionGV, MotionLLaMA, FineDance, ChoreoMaster,
OpenDance, PerMo are not on HuggingFace at all". They are -- inside OMG-Data. That error is
what made retargeting look necessary.

### 17.2 Four-language token gate: Telugu is binding

MuRIL fertility vs t5, measured on 6,000 parallel caption pairs per language from
`g1_dataset_robot_v2.csv`:

| language | script | median fertility | MuRIL > 50 | English gate |
|---|---|---|---|---|
| Bengali | Bengali | **1.000** | 0.47% | <= 50 t5 tokens |
| Tamil | Tamil | 1.095 | 0.70% | <= 46 |
| Hindi | Devanagari | 1.158 | 1.80% | <= 43 |
| **Telugu** | Telugu | **1.278** | 2.00% | **<= 39** |

**The all-four gate is 39 t5 tokens, set by Telugu.** Bengali is free (1.000 -- MuRIL
tokenizes it as efficiently as t5 tokenizes English). Every caption dropped is dropped
because of Telugu; relaxing to hi/bn/ta would recover ~8% more data.

**Corrects the earlier record**: SSOT s9.2 has Hindi fertility at 1.25 (measured on
bones-seed). On this corpus it is 1.158. **Fertility is corpus-dependent, not a property of
the language pair** -- re-measure per corpus, never inherit.

### 17.3 The corpus choice: Option D

Every candidate, scored at the 39-token gate with union-find groups over source + caption:

| option | episodes | 4-lang safe | motions | **leak-safe groups** | motion type |
|---|---|---|---|---|---|
| humanml only | 25,727 | 24,995 | 25,727 | 16,997 | capture |
| motiongv only | 537,676 | 405,108 | 483,017 | 465,945 | **authored** |
| omomo+100style+motionllama | 30,611 | 30,277 | 10,165 | 8,815 | capture |
| **D: humanml+omomo+100style+motionllama** | **56,338** | **55,286** | 35,892 | **25,811** | capture |
| amass+humanml | 64,746 | 63,832 | 32,034 | 21,165 | capture |

**CHOSEN: Option D.** Four whole datasets, no fractional sampling, ~50K target met.

**MotionGV rejected for Run 3 despite being 18x larger.** Rendered 10 episodes and measured
against real AMASS:

| | pelvis min | foot min | foot median | frames with foot < 5 cm |
|---|---|---|---|---|
| AMASS (real mocap) | 0.752 m | 0.048 m | 0.063 m | **11.6%** |
| MotionGV | 0.721 m | 0.092 m | 0.115 m | **0.0%** |

**MotionGV never makes ground contact** -- zero frames in ten episodes. Captions are
animation asset names with descriptions appended (*"The man is doing A Monster Powers Up."*,
*"Bizarre soldier"*, *"Hefterschwert"*), and 483,017 unique motions is 40x larger than AMASS,
the largest real mocap collection. It is a game-animation library. Contact-based metrics
return exactly 0.000 on it -- the same degenerate signature as SSOT #21. Keep as a possible
Stage-C scale-up: it is 54% of OMG's pretraining, so it matches the frozen decoder.

**AMASS rejected**: 69% overlap with Run 2 AND fewer independent groups than Option D
(21,165 vs 25,811) despite 15% more episodes -- 6.2 segments per motion collapse under
union-find. Count groups, not rows.

**The humanml 96% overlap with Run 2 is deliberate.** The Run-3 variable is LANGUAGES.
Holding the motion distribution near Run 2 keeps the multilingual result directly comparable
to the +10.46% Hindi baseline. Changing corpus and language count together would confound
them. Overlap would only be contamination with reused weights or a reused test set; Run 3
does neither.

### 17.4 Mirror counterfactuals: paired supervision, NOT augmentation

**Mirrors must never be added as independent training rows.** Three failures:

1. **They defeat the leakage gate silently.** `X` and `X_M` have a different `source_id` AND
   a different caption (side words swapped), so union-find matches on neither key. `X` in
   train and `X_M` in test would pass every check we have while putting a reflection of a
   training example in the test set.
2. **They contaminate the flip-rate metric.** A model that saw `X_M` in ordinary training is
   measured on memorisation, not counterfactual generalisation.
3. **They dissolve Contribution 2.** "We added flipped data and it learned flipping" is
   augmentation, not attribute grounding.

**MANDATORY FIX -- add a third union key:**

    uf.union(f"clip::{i}", f"mirror::{canonical_pair_id}")

so `X` and `X_M` are one super-group and always share a split.

**Mirrors are also not a scale mechanism.** Measured on the current corpus: 22.6% of clips
carry exactly one side word in both English and Hindi (2,997 of 13,282), projecting to
**~12,712 counterfactual pairs** in a 56,338-episode corpus -- **23x** the harness
requirement of n >= 550/arm, but only a 23% overlay, not a doubling. The corpus stays
**56,338 real motions**; counterfactual pairs sit on top.

Partition the pair set three ways, disjointly: **train** pairs feed the flip-consistency
loss, **val** pairs tune its weight, **test** pairs produce the reported flip rate on source
motions never seen in training.

### 17.5 The English adapter control arm *(required, not optional)*

Contribution 2's claim is that the multilingual path **surpasses** frozen English on
attribute grounding. Against frozen English alone (~4% flip rate) that claim does not
survive review, because the obvious objection is fatal:

> *"Your adapter received explicit counterfactual supervision. Frozen English received none."*

**Three arms, identical counterfactual supervision and identical pair partitions:**

| arm | supervision | role |
|---|---|---|
| frozen t5 English, no adapter | none | the backbone deficiency, ~4% flip |
| **English adapter + counterfactual** | identical | **the control that makes the claim survive** |
| Indic adapter + counterfactual | identical | the claim |

Infrastructure exists: Run 2 arms E/F are English adapters with the same architecture,
`norm_mode="corpus"`, and their own `adapter_calibration_en.pt`. Cost is one extra training
run in the same GPU class.

Both outcomes are publishable, and one of them is only reachable with the control:
* Indic >= English adapter -> **"surpasses English under identical supervision"**, the strong
  claim, now defensible.
* Both far above 4%, near each other -> **"attribute grounding is learnable in the adapter
  regardless of source language, and the deficiency is in the frozen backbone"** -- weaker
  but robust, and still fixes a measured universal deficiency.

**Design consequence:** the counterfactual train/val/test partition must be SHARED across the
English and Indic arms, so the flip rates are computed on the same held-out pairs.

---

## 18. Run 3 corpus — built and verified *(2026-09-07, all figures [M])*

### 18.1 Extraction complete

`tools/extract_omg_data.py` pulled Option D from `THU-MARS/OMG-Data` into the existing npz
contract. The already-validated converter and split builder run downstream unchanged; no new
decode path was added, deliberately (SSOT #14/#15/#21/#24 were all new-path-meets-old-path).

| | |
|---|---|
| episodes | **44,996** (0 duplicate clip_id) |
| unique source motions | **32,519** (1.4 segments each) |
| unique captions | 31,337 (1.44 episodes per caption) |
| hours | 86.8 |
| four-language gate survival | **97.7%** (44,996 of 46,052), vs 97.4% projected |
| rejected | bad_qpos 0, short 0, frame_mismatch 0 |
| OMG splits (pre-repair) | train 35,772 / val 4,716 / test 4,508 |

| dataset | episodes | motions | seg/motion |
|---|---|---|---|
| humanml | 24,347 | 24,347 | 1.0 |
| 100style | 10,260 | **810** | **12.7** |
| omomo | 6,102 | 4,180 | 1.5 |
| motionllama | 4,287 | 3,182 | 1.3 |

`source_amass` groups segments correctly -- `100style/Penguin_BW` collects all 46 of its
segments under one key, which is what the leakage split requires. **Expect 100style to
collapse hard under union-find**: 23% of rows, 2.5% of motion diversity. Report group count
alongside sample count.

### 18.2 The mirror union key is wired and PROVEN SENSITIVE

`build_amass_g1_splits.py` now takes a third union key. Verified with a synthetic corpus where
X and X_M carry different source ids AND side-swapped captions -- i.e. the real shape:

| test | result |
|---|---|
| T1 without the key: do pairs already group? | **0/4** -- the leak is real, the key is not redundant |
| T2 with the key: do pairs group? | 4/4 |
| T3 corrupted split: does the assertion fire? | FIRES |

**T1 is the test that mattered.** A gate that passes proves nothing until it has been shown to
fail on the thing it exists to catch.

### 18.3 Google Translate is not viable at this volume **[M]**

Measured against the live endpoint on 2026-09-07:

| spacing | success | throughput |
|---|---|---|
| 0.1 s | 45% | -- |
| 0.5 s | 65% | 0.49 cells/s |
| 1.5 s | 70% | 0.33 cells/s |

**Slowing down barely helps** -- there is a ~30% failure floor that is not rate-driven. At
0.49 cells/s one pass over a 60k-cell shard is ~34 h, so 2-4 days per person with re-passes.

**`deep_translator.translate_batch` is NOT a batch API.** `deep_translator/base.py:181` loops
calling `translate()` once per string: identical request count, no spacing, no retry, and one
failure aborts the batch. It is strictly worse under throttling. The earlier claim that it was
"~8x faster" was wrong and is corrected in the shipped doc.

**Cumulative per-IP blocking is real.** After a day of testing from one IP, a run produced
182 consecutive retries and **zero** successful translations. Not degraded -- refusing. The
fix is time, not configuration.

Two tool changes followed: `--loop` makes unattended passes until complete, and
`--abort-after` (default 60 consecutive failures with none succeeding) distinguishes *blocked*
from *throttled*. Without the latter, a fully blocked pass would grind ~250 h before the
end-of-pass no-progress check could fire, because each failed cell costs ~15 s of backoff.

**Open decision:** Option A (Google, 3 people, 2-4 days) is what the user chose over
IndicTrans2 on a rented GPU (~1-3 h, no rate limit) to preserve the manually verified output
quality. If two fresh IPs also collapse, that decision should be revisited -- the evidence
will be person 2 and person 3's first-pass rates.

---

## 15. Artifact and reference manifest

### 15.1 This repository

```
OMG/
├── MAILA_SSOT_2026-09-05.md              <- THIS DOCUMENT (single source of truth)
├── MAILA_RESEARCH_LOG_2026-08-22.md       Run 1 record; errors 1-8; still authoritative for Run 1
├── MAILA_OMG_EXP_PLAN.md                  original full plan (Hindi+Hinglish, E0-E4 tiers)
├── MAILA_OMG_EXP_PLAN_TRIMMED.md          the scoped plan Run 1 actually executed
├── DEEP_RESEARCH_PROMPT_2026-08-30.md     the leap-vs-package research brief
├── DEEP_RESEARCH_FINDINGS_2026-08-30.md   landscape verdicts, H1-H6, with arXiv IDs
├── run_artifacts/{selection,history}.json  Run 1 selection table and training trace
├── splits_bones_seed_30k/                 Run 1 corpus splits (BONES-SEED)
├── splits_amass_g1_13k/                   *** RUN 2 CORPUS SPLITS ***
│   ├── train.csv  val.csv  test.csv  reserve.csv
│   └── split_report.json
├── side_by_side/                          Run 1 25-clip comparison renders
└── tools/
    ├── maila_encoder.py                   MurilAdapterEncoder + ResidualAdapter (unchanged)
    ├── maila_train.py                     Run 1 loop; Run 2 imports build()/native_loss() from it
    ├── convert_g1_npz_to_omg125.py        *** RUN 2 CONVERTER *** (fps gate, per-source npz,
    │                                        start_s slicing, round-trip abort, all caption slots)
    ├── maila_train_motionpivot.py         *** RUN 2 TRAINER *** (symmetric loss, RNG replay,
    │                                        L_null gate, text-induced-variance diagnostic)
    ├── build_amass_g1_splits.py           *** RUN 2 SPLIT BUILDER ***
    ├── build_bones_seed_splits.py         Run 1 split builder (methodology reference)
    ├── convert_bones_to_omg125.py         125-D conversion via OMG's own codec
    ├── finalize_run.py                    fixed-seed re-scoring + selection
    ├── text_sensitivity.py                the rel probe
    ├── three_arm_benchmark.py             the 3-arm benchmark
    ├── direction_minimal_pairs.py         direction swap test + word inventory
    ├── smoke_phase1.py                    13 Phase-1 checks
    ├── contract_probe.py                  cheap qk-norm contract probe
    └── render_side_by_side.py             PIL-composited comparison renderer
```

### 15.2 Data and model artifacts

| artifact | location |
|---|---|
| **Run 2 corpus** | `D:\HumanML3d\g1_dataset_robot_v2.csv` (13,282 clips) |
| Run 2 motion `.npz` | `AdiShingote/pragya-vla-g1-motion-dataset` (HF) |
| Retargeting pipeline doc | `D:\HumanML3d\HUMANML3D_TO_G1_PIPELINE.md` |
| Run 1 artifacts | `CodeSushh/PragyaVLA-omgdit-runs/maila-hi-20260822-100455/` (HF, private) |
| Run 1 converted data | `CodeSushh/bones-seed-omg125-maila` (HF, private) |
| OMG weights / data | `THU-MARS/OMG` · `THU-MARS/OMG-Data` |
| BONES-SEED official | `bones-studio/seed` → `metadata/seed_metadata_v004.csv` |
| HumanML3D texts | `D:\HumanML3d\HumanML3D\HumanML3D\texts.zip` |
| SnapMoGen captions | `Ericguo5513/SnapMoGen` → `all_caption_clean.json` |
| Motion-X++ labels | `YuhongZhang/Motion-Xplusplus` → `text/semantic_label/*.zip` |
| MuRIL | `google/muril-base-cased` |
| MT control | `ai4bharat/indictrans2-indic-en-1B` |

### 15.3 Literature

**Directly built on:** OMG arXiv:2606.10340 · MuRIL arXiv:2103.10730 · DiT (Peebles & Xie, ICCV 2023)
· 6-D rotation (Zhou et al., CVPR 2019) · TMR arXiv:2305.00976 · FID (Heusel et al., NeurIPS 2017) ·
DDIM arXiv:2010.02502 · RoPE arXiv:2104.09864.

**The design family we are generalizing:** PEA-Diffusion arXiv:2311.17086 (ECCV 2024 — output-space
KD from an English teacher; the asymmetric special case of our objective) · AltDiffusion
arXiv:2308.09991 (AAAI 2024) · MuLan arXiv:2412.01271.

**Nearest competitor:** BiMD / BiHumanML3D arXiv:2603.25178 — text-space alignment via a
vision-language teacher, en–zh, SMPL space. Also the annotation-quality bar.

**Why the language axis matters (third-party evidence):** *Beyond English* arXiv:2606.15714 ·
*When Does Language Matter?* arXiv:2606.11906.

**The crowded English field (what we must not claim):** UH-1 arXiv:2412.14172 · Humanoid-LLA
arXiv:2511.22963 · FRoM-W1 arXiv:2601.12799 · TextOp arXiv:2602.07439 · DAJI arXiv:2605.14417 ·
LeVERB arXiv:2506.13751 · UniAct arXiv:2512.24321 · GR00T N1 arXiv:2503.14734.

**Methodology:** Artetxe et al., *Translation Artifacts in Cross-lingual Transfer Learning*,
EMNLP 2020 — why native-authored evaluation is non-negotiable.

**Recorded so nobody re-cites it as settled:** ReAlign arXiv:2505.04974 — **withdrawn by its authors
2025-08-01**.

---

## 16. The approach in one paragraph

> Train one shared adapter over Hindi, Bengali, Tamil and Telugu between a frozen MuRIL and a frozen
> OMG-100M, replacing the English-teacher response loss with **cross-lingual response consistency**:
> holding motion, history, timestep and noise fixed via RNG replay, the predicted clean motion under
> language A must match the prediction under language B. Motion is the pivot — no text-space
> distillation, no vision-language teacher, no language privileged as the semantic authority (a small
> English anchor is retained only because t5-base geometry is where the frozen cross-attention was
> fitted). Guard the objective's degenerate solution with a physically-exact mirror-counterfactual
> term and a non-optimized collapse diagnostic. Then hold one language out entirely and measure
> zero-shot transfer through the frozen motion prior, within and across the Indo-Aryan/Dravidian
> boundary, as a few-shot curve. Evaluate on the first native-verified multilingual benchmark for
> robot-executable motion.
