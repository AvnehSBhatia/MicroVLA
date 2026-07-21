# MicroVLA v2

A micro vision-language-action (VLA) pipeline: a single frozen off-the-shelf
detector — YOLO-World-S — supplies both open-vocabulary vision *and*
language grounding (its own internal CLIP text tower), feeding a set of
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
`cfg.real_frame_hz = 2` Hz, producing grounded source/target boxes. The
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
