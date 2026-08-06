# MicroVLA

**A ~30M-parameter vision-language-action stack with unaided pick-and-place
success on LIBERO — and a reproducible audit showing how fixed-placement
benchmarks let non-visual policies score, plus the small repair that fixes it.**

> **Audit result, 2026-08-06.** Placement memorization was found at **four**
> layers: the benchmark, the selection loop, the grasp head — and, found last
> by swapping the instruction, **the place head**, which learned a
> command→location map rather than the basket. Against a basket that moves
> 0.22–0.40 cm across all ten tasks, the released head emits the *correct*
> point for the command it trained on (0.36 cm error) and points **14.5 cm**
> and **13.0 cm** away for two it did not. A head trained on all three commands
> collapses that spread to 0.78 cm, so the failure is **training coverage, not
> architecture**. This layer was invisible to every earlier probe because all
> ten tasks share one basket — memorized and grounded place maps are
> behaviourally identical until the instruction is swapped.

## Release: structured-decoding policy + de-memorization results (2026-08)

| protocol (task 0, no assist flags, all successes filmed) | memorized head | **released head (v5)** |
|---|---|---|
| held-out init states (never used in tuning) | 0.300 | **0.700** (n=10; confirmed n=50: 35/50 [0.56, 0.81]) |
| randomized placements (±4 cm teleports) | **0.100** (control, identical draws) | **0.400** (n=10; n=50: 26/50 = 0.520 [0.39, 0.65]) |
| input attribution | eef-tracking, image-flat | **visual** (vision 1.1–1.8 cm, proprio 0.1 cm) |
| survives detector-stack rebuild | dev 7/10 → **2/10** | **all three protocols reproduce** |

**Multi-object addendum (dated).** Two-object head (`goal_heads_v7.pt`)
under the anchor trust-region latch — one config, no per-object
constants: pre-registered fresh-seed confirmation **butter 5/10 + soup
3/10** (selection-band 6/10 + 4/10; butter reaches **10/10** under the
early-latch config split). Three-object head (`goal_heads_v8.pt`, adds a
cream-cheese corpus) with semantic rebinding: selection-band butter 7/10
+ soup 5/10, second pre-registered fresh-seed confirmation **butter 5/10
+ soup 4/10**.

**Cream cheese: first successes, 2026-08-06.** Cream was 0/10 under every
configuration, binder and prompt fix in this campaign. Nine mechanisms were
proposed for it and **eight refuted by measurement**; the ninth —
appearance-side off-manifold drift — survived falsification, a circularity
check, and an intervention. Fine-tuning the grasp head on the machine's *own*
deployment viewpoints (labelled from the simulator's object pose, which the
teacher corpus cannot do for viewpoints it never visited) cut cream's deployed
lateral error **56%** (0.0697→0.0310 m, Mann-Whitney *p*=0.020) and produced
its **first two successes** (2/10 on held-out seed 20), while soup went 3/6 →
7/10 — the repair costs nothing. Collection used seed 0, evaluation seed 20, so
no cell is scored on its own band. We do **not** claim cream is crossed: 2/10
vs 0/10 is Fisher *p*=0.474. Films in `watch_videos/dagger_cream/`.

**Corrected 2026-08-06.** This paragraph previously said cream fails because
the detector's role binding "cannot separate it from look-alikes". That was
measured and **retracted**: role binding separates *nothing*. Running two
objects' prompt chains over the same frames returns the **same detection**
(same-box rate 0.87–1.00, median centre distance 0.0001) — every product name
scores 0.00 on the region-text head, so all chains fall through to a generic
"box", **including the chains of the two objects that do cross**. Soup
succeeds at 35/50 while being indiscriminable in exactly the sense cream was
blamed for. Deployed binding is therefore **identity-blind**: soup's success
is not explained by binding the named object, and the repaired head is
grounded on *a* box rather than *the named* one. Cream's own failure was
later traced elsewhere entirely: a world-space probe shows it reaches the
**commanded** object 6/6 and fails by a systematic ~7 cm *lateral* grasp
error, so binding was never its bottleneck (see the first-successes note
above). The 0.902 / 0.613 figures
are *offline* identity accuracy on corpus crops and dissociate from deployed
success; binder cells are dated in App D.

**Identity-blind is not language-blind.** Swapping the instruction to a
different object — env, physics and success criterion unchanged, so success is
still scored on the real task — collapses the cell from **7/10 to 0/10** (told
butter; 0/10 again told cream cheese; exact two-sided *p* = 0.016). The system
is strongly instruction-sensitive, and a pre-registered prediction that the
swap would be inert is **falsified**. Telemetry places the butter cell's
failure *downstream of object approach* — the machine still reaches the true
soup object as closely as baseline. A 2×2 decomposition (driving the detection
prompts and the place head's command embedding from *different* instructions)
localises the collapse **entirely to the embedding channel**:

| | embeddings = soup | embeddings = butter |
|---|---|---|
| **prompts = soup** | **7/10** | **0/10** |
| **prompts = butter** | **6/10** | **0/10** |

Exact McNemar: prompt swap vs baseline *p* = 1.0000; embedding swap vs baseline
*p* = 0.0156; embedding swap vs the full swap — **identical patterns**. With the
architecture (`goal_machine.step(proprio)` replaces the plan; the grasp head
takes no text; `set_place(place_head(command_emb))` runs once per episode),
**object selection, approach and grasping run with no language input at all,
and the whole language channel is a single latched (x, y) place point.** Ask
this policy for the butter and it finds, approaches and grasps the alphabet
soup at baseline rate. Full record in `paper/paper.md`.

The free-regression baseline (same trunk, six controlled variants) is 0.000;
structured decoding — two learned goal heads (0.24M) driving a task-content-free
servo shell — is the difference in kind. Full evidence chain in
[`paper/MANUSCRIPT.md`](paper/MANUSCRIPT.md) (results §6.4d–f), the complete lab
log in [`paper/paper.md`](paper/paper.md), scoreboard in
[`results/UNAIDED_LEADERBOARD.md`](results/UNAIDED_LEADERBOARD.md), and demo
films in [`demo/`](demo/).

### Model files (`models/`)

| file | role | params |
|---|---|---|
| `full_stageB_rec_fix.pt` | trunk: fusion/drift/relational/planner + TRM (frozen at deploy) | ~17M trained-side |
| `goal_heads_v5.pt` | structured-decoding goal heads (grasp + place), variance-trained + jitter | 0.24M |
| `goal_heads_v7.pt` | two-object heads (soup + butter corpus, 2026-08-05 addendum) | 0.24M |
| `goal_heads_v8.pt` | **three-object heads** (soup + butter + cream corpus, 2026-08-06 addendum) | 0.24M |
| `role_bank.pt` | per-object crop-embedding banks for 1-NN role binding (`--goal-src-bank`); measured 0.902 identity accuracy vs 0.613 for a mean prototype | 1432 vectors |
| `role_prototypes.pt` | mean-prototype variant of the same (`--goal-src-proto`); shipped because its **negative** result is part of the evidence | 3 vectors |
| `gates_v1.pt` | stage-1 learned gates (close trigger + hold check) | <3K |
| `goal_heads_dagger.pt` | v8 fine-tuned on the machine's **own** deployment viewpoints (DAgger, collected seed 0). Cream's first successes: 2/10 held-out, lateral error −56% (*p*=0.020); soup 3/6→7/10. Checkpoint meta records the collection band and label derivation | 0.24M |

Both binders are built from the training corpora by
`scripts/build_role_bank.py` / `scripts/build_prototypes.py` — no new
episodes and no gradient steps, but corpus-derived, so they are part of
the learned system rather than a free lunch.

(The frozen YOLO-World-S detector downloads automatically via `ultralytics`.)

### Run it

Environment (Linux + CUDA; python 3.10 venv — 3.12 cannot run the sim stack):

```bash
python3.10 -m venv .venv310 && . .venv310/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy ultralytics opencv-python-headless h5py easydict cloudpickle \
    "gym==0.26.2" imageio termcolor hydra-core bddl future transformers pytest \
    einops matplotlib pillow robosuite==1.4.1 mujoco==2.3.7
git clone https://github.com/Lifelong-Robot-Learning/LIBERO .libero_src   # full clone
export PYTHONPATH=$PWD/.libero_src:$PWD  MUJOCO_GL=osmesa
```

Unaided eval (held-out protocol, per-success videos):

```bash
python -m eval.libero_eval --suite libero_object --task-ids 0 --n-trials 10 \
  --max-steps 600 --seed 20 \
  --checkpoint models/full_stageB_rec_fix.pt --norm-stats eval/identity_norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 --perception-period 2 \
  --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --goal-ckpt models/goal_heads_v5.pt \
  --success-video-dir out_videos --out-dir out_results
```

Add `--randomize-source-xy 0.04` for the randomized-placement protocol,
`--gates-ckpt models/gates_v1.pt` for the learned-gates variant, and
`--mock-env --checkpoint none` for a no-sim smoke test. CPU-only unit tests:
`python -m pytest tests -q` (600+, no sim required).

**Honest scope** (details in MANUSCRIPT §9): task content (where to grasp/place)
is fully learned; the control shell is engineered and task-free; results are
single-object (soup) — the per-object teacher campaign for multi-object
generalization is documented ongoing work.

---

# MicroVLA v2

A micro vision-language-action (VLA) pipeline: a single frozen off-the-shelf
detector — YOLO-World-S — supplies both open-vocabulary vision *and* the only
text encoding in the stack (its own internal CLIP text tower), feeding a set of
small, novel trainable heads — hard-capped at **9M trainable parameters
total** — that turn "task text + video stream" into normalized 7-servo PWM
plans. A 30 Hz JEPA-style latent rollout runs real perception at 2 Hz and
fills the other 14-of-every-15 ticks with a corrected world-model
prediction, so the control loop stays fast even though the detector doesn't.
A ~10M-param Tiny Recursive Model (TRM) sits in the middle of the stack —
interface and spec in `microvla/trm/`, real implementation at the repo root
(`TRM.py::RecursiveTRM`, ~9.5M params, residual world model).

`DESIGN.md` is the binding architecture contract; this README is the tour.

## Architecture

```
                        ┌── once per task ─────────────────────────────────┐
"move can to ball" ──►  │ parse_command: source="can", target="ball"       │
                        │ YOLO-World CLIP text tower (via set_classes) ──► │
                        │ 3 ordered CLIP embs [512]: command, source, target
                        └──────────────────────────────┬───────────────────┘
                                                       │
camera 30 Hz ─┬─ every 15th tick (2 Hz) ─ REAL TICK ───▼───────────────────────────┐
              │   YOLO-World-S (frozen): frame_emb [512] (GAP of SPPF map)         │
              │     source box: emb [512] + center [2]                            │
              │     target box: emb [512] + center [2]   (per-class best box)     │
              └─ other 14 ticks ─── DREAM TICK ────────────────────────────────┐  │
                    frame token = corrected TRM prediction [512]               │  │
                    boxes HELD from last real tick, evidence weight            │  │
                    decayed by staleness (trained via modality_dropout         │  │
                    evidence fade — the same weighting continuum)              ▼  ▼
  SlotResonanceFusion: 32 slots cross-attend over 8 role-tagged tokens
      [cmd | src | tgt | frame | src-box | tgt-box | geometry | last action]
      (box/geometry tokens scaled by confidence × freshness)  ──► fused [32, 5]
                                                                        │
  AnchoredDriftEncoder (anchor = first REAL frame, GRU accum,
                        steps on REAL ticks only)             ──► state_delta [256]
                                                                        │
  TRM (TRM.py::RecursiveTRM, ~9.5M — residual world model)             ▼
      forward(fused [B,32,5], state_delta [B,256], current_emb [B,512])
        -> next_emb [B,512]  (= current + predicted change)
                     │
                     ├──► InnovationCorrector (Kalman-lite) ──► corrected latent → next tick
                     ▼
  ChronoQueryPlanner(next_emb [512]) ──► raw plan [5, 7] in [-1, 1]
      emitted plan = τ·raw + (1−τ)·previous plan (trust HOLD-blend)
      rows = 5 sequential timesteps (1/30 s apart; row 0 executed now,
      fed back as the action token), cols = 7 servos, normalized PWM
```

Two ways to run the stack:

* **`JEPALoop`** — the deployment path: 30 Hz ticks, real perception at
  2 Hz, dream ticks in between (see [JEPA at 30 Hz](#jepa-at-30-hz) below).
* **`MicroVLAPipeline`** — the simple 2 Hz real-only path (`step()` is
  exactly a JEPA real tick without the corrector). Handy for offline
  debugging and as the TRM builder's minimal harness.

## What is novel in each trainable module

**Slot Resonance Fusion** (`microvla/fusion/slot_fusion.py`, ≤5.0M params,
target ~4.5M). Eight role-tagged tokens — 3 text tokens (command, source,
target), the frame token, the source-box token, the target-box token, a
Fourier-encoded geometry token built from `[fourier(src_center),
fourier(tgt_center), fourier(tgt_center − src_center), box_weights]`, and an
action token carrying the previously executed servo command — are projected
to a shared `d_model=384` space with a learned per-position role embedding.
The COMMAND embedding FiLM-modulates (scale + shift) the frame and box
tokens, so *what* the robot was told to do reshapes how it looks at the
scene before attention even runs. 32 learned slot queries then run 3 rounds
of pre-LN multi-head cross-attention over the 8 tokens, and a shared
low-rank head compresses every slot down to 5 numbers — the tiny,
structured `[32, 5]` interface the TRM consumes.

The genuinely novel bit is **continuous evidence weighting shared between
training and dreaming.** Every box token (and the geometry token) is scaled
by `box_weight = confidence × freshness`: real ticks pass the detector's
confidence, dream ticks hold the last real boxes and decay their weight by
`staleness_decay^k` (objects don't teleport in 33 ms — zeroing them, as v2
did, threw away near-perfect information), and a genuinely missed detection
passes weight 0 — which also disambiguates the center-frame fallback from a
real object at frame center. Train-time `modality_dropout` fades the same
weights by a random factor, so by the time the JEPA loop dreams, the network
has been trained on the entire evidence-decay continuum it will actually
see. An eighth **action token** carries the previously executed servo
command (plan row 0), so the world model learns *controlled* dynamics — it
knows what the arm was just told to do.

**Anchored Drift Encoder** (`microvla/aux_state/drift_encoder.py`, ≤1.5M
params, ~0.73M). Rather than encoding absolute scene state, it encodes
*multi-timescale drift* against a **context window**: a rolling memory of
the last 8 real-frame embeddings plus the episode anchor (the first REAL
frame). Each step builds one drift token per reference — anchor, and lags
1/2/4/8 frames (≈0.5–4 s at 2 Hz) — from `[emb − ref, emb ⊙ ref]` with a
shared projection and learned horizon embeddings; a learned-query attention
pool reads the window, a sigmoid gate filters it, and a `GRUCell(256, 256)`
still accumulates context older than the window. The LayerNorm'd
`state_delta [256]` is "how the world has been moving, at every timescale
that matters," not just a first-vs-latest diff. It steps on REAL ticks only
— held constant across dream ticks — so the summary integrates measured
evidence, never accumulated imagination. The TRM additionally receives its
own **latent context window** (the last 8 tick latents, compressed by two
learned fast/slow decay profiles inside `RecursiveTRM`), so the world model
sees recent trajectory, not just the current instant.

**Chrono-Query Planner** (`microvla/planner/chrono_planner.py`, ≤2.5M
params, target ~1.6M). The predicted next-frame embedding is reshaped into 8
memory tokens of width 64, projected to `d_plan=256`; 5 learned time-query
tokens carrying a fixed sinusoidal step encoding cross-attend over that
memory for 3 rounds. Crucially the head predicts per-step **deltas**, and
the plan is `tanh(cumsum(deltas, dim=1))` — smoothness and sequential
consistency are built into the decoding itself, not just penalized by a
training loss.

**Innovation Corrector** (`microvla/jepa/corrector.py`, 0 learned params).
A Kalman-lite complementary filter that is the glue making dream ticks safe.
On every real measurement it computes the innovation
`e = real_emb − pending_pred` and EMAs it into a correction vector
`c ← β·c + (1−β)·e`, and sets a **self-calibrating** trust score from the
error *ratio*: `τ = exp(−½·(‖e‖/err_bar)²·temp/4)`, where `err_bar` is an
EMA of recent innovation norms. There is deliberately no fixed cosine
threshold — standardized frame embeddings of a near-static scene are always
highly correlated, so absolute-cosine trust would saturate; instead the TRM
is compared against its *own recent accuracy*. Each dream tick applies a
*decaying* fraction of the correction, `pred + decay^k · c` (then
re-standardized into the canonical space). Low trust **hold-blends** the
plan — `τ·new + (1−τ)·previous` — freezing current commands rather than
scaling absolute PWM targets toward the mid-range pose (which would be a
real, possibly large, commanded motion).

## JEPA at 30 Hz

The control loop (`microvla/jepa/loop.py`) ticks at `cfg.tick_hz = 30`. Every
`round(tick_hz / real_frame_hz) = 15`th tick (`0, 15, 30, 45, ...`) is a
**real tick**: YOLO-World-S actually runs on the camera frame at
`cfg.real_frame_hz = 2` Hz, producing source/target boxes. (Measured
2026-08-06: those boxes are *not* identity-bound — see the correction above.
"Source" names the role the box plays in the servo shell, not a verified
identity match to the instruction's noun.) The
other 14 of every 15 ticks are **dream ticks**: no frame is consumed; the
corrected (re-standardized) TRM prediction from the previous tick becomes
the frame token, the last real boxes are held with staleness-decayed
evidence weights, and a new prediction is produced. Story: **2
Hz real perception, 28 Hz latent dreaming** — the servo plan updates at the
full 30 Hz tick rate even though the camera/detector only contributes once
every half second.

**Why this is a reasonable compute trade, not a hack:** YOLO-World-S is a
~13M-parameter convolutional detector run over a full camera frame — by far
the most expensive op in the stack, and it only runs at 2 Hz. A dream tick
runs only the trainable heads plus the TRM: `SlotResonanceFusion +
AnchoredDriftEncoder + ChronoQueryPlanner + TRM ≈ 4.5M + 0.9M + 1.6M + 10M
≈ 17M` params of small attention blocks and a GRU cell operating on
`[32, 5]`/`[256]`/`[512]`-sized tensors — no image ever touches them. That
combination is light enough to run at 28 Hz on CPU, which is what makes the
30 Hz plan-update rate achievable without a GPU, while the InnovationCorrector
keeps the 14-tick-long open-loop stretches from drifting unchecked.

## The v2 parameter ledger

| item | budget |
|---|---:|
| YOLO-World-S detector (frozen, resident at runtime) | ~13M |
| CLIP text tower (separate ~63M model; runs ONCE per task at `set_classes`, precomputable offline — NOT resident on-device) | 0 resident |
| TRM (open slot, reserved) | 10M |
| **Trainable heads total (hard cap `cfg.trainable_param_budget`)** | **9M** |
| — SlotResonanceFusion | ≤5.0M (target ~4.5M) |
| — AnchoredDriftEncoder | ≤1.5M (target ~0.9M) |
| — ChronoQueryPlanner | ≤2.5M (target ~1.6M) |
| InnovationCorrector | 0 (no learned params) |

MiniLM is gone in v2 — text comes from YOLO-World's own CLIP text tower, so
there's no separate ~22.7M language encoder to carry. Total deployed ≈
13 + 10 + ~7 ≈ 30M, under the 32M envelope. Run the audit yourself:

```bash
python -m microvla.utils.param_audit
```

`tests/test_param_budget.py` enforces the same caps in CI.

## The TRM slot (handoff)

The TRM predicts the *next* frame embedding (residually, on top of the
current one) from the fused task/perception matrix, the drift code, and the
current latent. Contract, 10M param budget, **FLOPs budget**, recommended
architecture, and (documented-only) training loss live in
[`microvla/trm/TRM_SPEC.md`](microvla/trm/TRM_SPEC.md) — read the
"CONTRACT CHANGE (v3)" box first. The real implementation already exists at
the repo root: `TRM.py::RecursiveTRM` (~9.5M params, weight-tied recursion,
FiLM drift conditioning, single-pass inference). Wire it in with:

```python
from microvla import JEPALoop, DEFAULT_CONFIG
from TRM import RecursiveTRM

loop = JEPALoop.build_real(DEFAULT_CONFIG, trm=RecursiveTRM(DEFAULT_CONFIG))
loop.set_task("move can to ball")
```

Any alternative implementation just subclasses `TRMBase`:

```python
class MyTRM(TRMBase):
    def forward(self, fused, state_delta, current_emb):
        # fused [B,32,5], state_delta [B,256], current_emb [B,512]
        # -> next_emb [B,512]  (return current_emb + predicted_delta)
        ...
```

If no TRM is passed, `build_real` logs a warning and falls back to the
`MockTRM` stub (a single `Linear(416, 512)`, ~0.21M params) so the loop
still runs end-to-end.

### TRM training loss (documented, NOT implemented)

No TRM training code exists in this repository. The documented loss
(`train.losses.trm_loss_documentation()`, authoritative version in
`microvla/trm/TRM_SPEC.md`): the predicted `next_emb` is regressed onto the
*actual* YOLO frame embedding of the next REAL frame with

```
L = 1.0 * (1 - cosine(y_hat, y)) + 0.5 * MSE(y_hat, y)   # on LayerNorm-standardized targets
```

plus an optional in-batch InfoNCE term, with an EMA/stop-grad target-encoder
note in case the YOLO backbone is ever fine-tuned (collapse risk). Multi-step
rollout training is **mandatory**, not optional: at inference the TRM runs
~14-step open-loop dream rollouts between real measurements, with each
prediction fed back through fusion's dream path exactly as the JEPA loop
does. Training must reproduce that feedback loop with a scheduled horizon
`H` (start at 1, grow to 14) and a discounted loss `sum_h 0.95^h * L_h` —
single-step-only training will compound error the InnovationCorrector alone
cannot save.

## Probing *your own* policy's language grounding

`eval/probes.py` ships the two annotation-free probes that found the results
above. They need no labels, no simulator and no second model — only a
perception object with `set_role_prompts`/`perceive`, or paired success
vectors. Both apply to any detector-grounded policy, not just this one.

```python
from eval.probes import prompt_agreement, instruction_swap

# 1. Does your grounding stage IGNORE the instruction?
#    Run two objects' real deployed prompt chains over the SAME frames.
r = prompt_agreement(perception,
                     chain_a=["alphabet soup", "soup", "box"],
                     chain_b=["butter", "box"],
                     frames=real_frames)
print(r.verdict())   # IDENTITY-BLIND / DISCRIMINATING / MIXED / INCONCLUSIVE

# 2. Does some stage MEMORIZE it? Run a task while telling the policy to
#    fetch a different object; keep the env and success criterion on the
#    real task, and pair the trials (same seeds, same init states).
s = instruction_swap(baseline=[...], swapped=[...])
print(s.verdict())   # exact McNemar, and it prints its own power ceiling
```

They fail differently on purpose: the first catches a stage that ignores the
instruction, the second a stage that memorized it. The second is the only way
to see a memorized command→location map on a benchmark where every task shares
one target — there, memorized and grounded maps are behaviourally identical
until the command changes. In this repo the swap arm is
`eval/libero_eval.py --override-instruction` (plus `--override-prompt-only` to
drive the prompt and embedding channels apart).

## Quickstart

Commands below assume you're at the repo root.

```bash
# Core install (torch + numpy only; mock pipeline, JEPA loop, tests, and
# training scaffold all work with this alone)
pip install -e ".[dev]"

# Run the test suite (CPU-only, mocks, no downloads, no cv2)
pytest

# Audit the v2 parameter ledger (asserts the 9M cap + per-module caps)
python -m microvla.utils.param_audit

# Smoke-train the heads on synthetic episodes (CPU, checkpoints -> ./checkpoints/)
python train/train_planner.py
# ...with the dream/modality-dropout path exercised at a non-default rate:
python train/train_planner.py --modality-dropout 0.5
```

Real inference with YOLO-World weights (installs the heavy perception stack;
`yolov8s-worldv2.pt` downloads on first use):

```bash
pip install -e ".[perception]"
```

```python
from microvla import MicroVLAPipeline

# 2 Hz real-only path: simplest way to sanity-check real perception.
pipe = MicroVLAPipeline.build_real(device="cpu")  # add trm=MyTRM(cfg) when ready
results = pipe.run("demo.mp4", "pick up the red block", max_steps=20)
for r in results:
    print(r.plan)  # [5, 7] normalized PWM targets in [-1, 1]
```

```python
from microvla import JEPALoop

# 30 Hz deployment path: real YOLO perception at 2 Hz, TRM-driven dream
# ticks fill the other 14 of every 15 ticks.
loop = JEPALoop.build_real(device="cpu")  # add trm=MyTRM(cfg) when ready
results = loop.run(camera_frames_at_30hz, "pick up the red block")
for tick in results:
    print(tick.is_real, tick.trust, tick.plan)  # plan already trust-scaled
```

## Data: BridgeData V2 (pretrain) + LIBERO (fine-tune/eval)

`preprocess/` converts both datasets into MicroVLA's `.npz` episode format
offline — the frozen perception stack runs once at conversion time, so
**training never touches images** and episodes are ~1000× smaller than raw
video. Both datasets are 7-DoF (Δxyz, Δrpy, gripper) = `num_servos=7`; frames
are subsampled to 2 Hz with the same cadence as the online sampler; actions
are quantile-normalized (`norm_stats.json` — keep it with checkpoints) and
chunked into the 5-row plan windows. Optional **TinyVLA teacher distillation**
(`--teacher tinyvla`) relabels actions with a pretrained VLA, cached per
episode. Nothing is downloaded automatically. Full guide:
[`preprocess/README.md`](preprocess/README.md).

```bash
# Disk-capped workflow (hard 10 GB budget, incl. downloads): stream shards —
# download one, convert, delete raw, repeat — with a hard usage guard.
python -m preprocess.shard_pipeline shards.txt data/bridge --dataset bridge --budget-gb 10 --device mps

# Or, with local copies already on disk:
python -m preprocess.bridge /data/bridgedata_raw data/bridge          # pretrain set
python -m preprocess.libero /data/libero_object  data/libero_object   # fine-tune/eval

python train/train_planner.py --data-dir data/bridge --device auto    # MPS on Apple silicon
```

## Evaluation

`eval/` closes the loop: `eval/policy.py::MicroVLAPolicy` wraps `JEPALoop` into
a duck-typed `reset(instruction)` / `act(frame_rgb) -> action` policy (owning
the `perception_period` real/dream schedule itself, independent of
`cfg.tick_hz`/`cfg.real_frame_hz`, since that schedule is the E4 sweep knob),
`eval/libero_eval.py::run_eval` drives it through a LIBERO suite (real sim,
or the dependency-free `MockLiberoEnv` via `--mock-env`), `eval/sweep.py`
runs the full perception-rate x {ours, persistence, linear} grid (paper.md
E4/E5, with the τ→failure AUROC folded in for free), and `eval/scorecard.py`
scores a checkpoint offline against a converted dataset's val split
(rollout error vs persistence, innovation norms, planner BC loss — no env
needed).

```bash
# Closed-loop LIBERO eval, one suite, real checkpoint:
python -m eval.libero_eval --checkpoint checkpoints/full_stageB.pt \
    --norm-stats data/libero/norm_stats.json --suite libero_object

# Same, but dependency-free (no LIBERO/robosuite install, no sim, no
# network) -- this is what CI runs; `--checkpoint none` also skips loading
# trained weights for a pure harness smoke test:
python -m eval.libero_eval --mock-env --checkpoint none --suite libero_object --n-trials 3

# The perception-rate sweep (the paper's central result) against the mock env:
python -m eval.sweep --mock-env --checkpoint none --periods 1 5 15 --n-trials 3

# Offline scorecard: rollout error / innovation norms / BC loss, no sim at all:
python eval/scorecard.py --checkpoint checkpoints/full_stageB.pt --data-dir data/libero
```

`eval_results/` (gitignored) collects the JSONL telemetry and JSON summaries
from all three.

## Repo layout

```
microvla/
  config.py                    # canonical dims (single source of truth)
  pipeline.py                  # MicroVLAPipeline + StepResult (2 Hz real-only path)
  perception/                  # command_parser, video sampler, CLIP task encoder, YOLO-World (+ mocks)
  fusion/                      # SlotResonanceFusion
  aux_state/                   # AnchoredDriftEncoder
  trm/                         # TRMBase, MockTRM, TRM_SPEC.md (open slot)
  jepa/                        # JEPALoop, InnovationCorrector (30 Hz deployment path)
  planner/                     # ChronoQueryPlanner
  utils/param_audit.py         # v2 ledger + budget assertion
train/                         # losses, EpisodeDataset, BC training scaffold
preprocess/                    # LIBERO + BridgeData V2 -> .npz converters, TinyVLA teacher
tests/                         # CPU-only, mock-only pytest suite
```

## Evaluation targets (paper / demo goals)

The claims this system is built to demonstrate, in the order they should be
proven:

1. **Task success at micro scale.** Success rate on a defined pick/push task
   suite (sim first — e.g. PushT-style or a Meta-World subset — then a real
   7-servo rig), with the whole deployed stack ≈ 30M params. Baselines:
   plain 2 Hz behavior cloning (no world model) and a quantized large-VLA
   teacher, if distillation is used.
2. **The rollout ablation table** (this is the scientific core):
   corrector on/off, dream-training (`modality_dropout > 0`) on/off,
   TRM rollout horizon 1 vs 14, fused matrix 8x5 vs 32x5. Each row is a
   success-rate delta that shows the corresponding design choice carries
   real weight.
3. **Edge latency/energy.** Per-tick latency at 30 Hz and per-frame YOLO
   latency at 2 Hz on a Raspberry Pi 5 (int8 perception at 416px, optional
   Hailo AI HAT), plus watts — against any published small VLA on the same
   board.
4. **Trust telemetry.** Corrector trust (tau) correlating with actual task
   failure — evidence the system knows when its imagination has diverged
   (and, via trust-scaled plans, acts conservatively when it does).
