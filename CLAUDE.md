# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Micro vision-language-action stack (~30M params deployed) targeting a Raspberry Pi 5 with a
7-servo rig. `DESIGN.md` is the **binding architecture contract** — read it before changing
any module interface, and update it in the same change if an interface must move.
`microvla/config.py` (`MicroVLAConfig`) is the single source of truth for every dimension;
never hardcode a dim that exists there.

## Commands

```bash
.venv/bin/python -m pytest tests -q            # full suite (CPU-only, mocks, no network)
.venv/bin/python -m pytest tests/test_jepa_loop.py -q                  # one file
.venv/bin/python -m pytest tests/test_shapes.py::TestChronoQueryPlanner -q  # one class/test (-k works too)
.venv/bin/python -m microvla.utils.param_audit # asserts the 9M cap + per-module caps
.venv/bin/python train/train_planner.py --epochs 2 --episodes 4   # smoke train
```

Fresh setup if `.venv` is missing: `python3 -m venv .venv && .venv/bin/pip install torch numpy pytest`
(or `pip install -e ".[dev]"`). There is no linter configured.

The venv at `.venv` has only `torch`, `numpy`, `pytest`. Keep it that way: `import microvla`
must always succeed with torch+numpy alone. Heavy deps (`ultralytics`, `cv2`, `torchvision`)
are imported lazily inside the classes that need them (`pip install -e ".[perception]"`).

## Architecture in one paragraph

Text is parsed (`perception/command_parser.py`) into ordered source/target phrases; CLIP
embeddings for (command, source, target) are harvested from YOLO-World's own text tower once
per task (`ClipTaskEncoder`) — there is no separate text model. At 2 Hz, frozen YOLO-World-S
supplies a frame embedding plus per-role box embeddings/centers (SPPF hook + ROIAlign).
`SlotResonanceFusion` → `[B, 32, 5]`; `AnchoredDriftEncoder` (anchored on the episode's first
real frame) → `[B, 256]`; both feed the TRM, which predicts the next frame embedding
`[B, 512]`; `ChronoQueryPlanner` decodes it to a `[B, 5, 7]` plan (5 sequential updates,
7 servos, tanh-bounded). `JEPALoop` runs this at 30 Hz: real perception every 15th tick,
the other 14 are dream ticks feeding the corrected TRM prediction back through fusion's
dream path, with `InnovationCorrector` (no params) doing drift correction and trust-scaling
the emitted plan.

## Hard rules

- **The TRM package is an interface-only slot; the real TRM lives at root `TRM.py`.**
  `microvla/trm/` contains only the `TRMBase` interface, `MockTRM`, and `TRM_SPEC.md` —
  keep it that way. The real implementation is `TRM.py::RecursiveTRM` (~9.5M params,
  weight-tied recursion + FiLM drift conditioning, deliberately outside the package;
  maintained by a collaborator — coordinate before restructuring it). v3 contract:
  `(fused [B,32,5], state_delta [B,256], current_emb [B,512], context=None [B,K,512])
  -> [B,512]` with the **residual convention** (`return current_emb + delta`); the
  `context` window of recent tick latents is caller-owned state (the loop maintains
  it) — the TRM itself must stay stateless; plug in via
  `JEPALoop.build_real(trm=RecursiveTRM(cfg))`. `forward()` runs ONE refinement pass
  (`n_sup_infer=1`) — the deep-supervision passes are training-only via
  `refine_forward` + `TRM.py::spec_loss` (cosine + raw MSE, no LayerNorm in the loss).
  `python TRM.py` runs its self-test. `train/` still contains no TRM training code.
- **Canonical embedding space.** Perception standardizes every visual embedding
  (zero mean / unit std per vector, `microvla/utils/embedding.py::standardize`) at the
  boundary; the JEPA loop re-standardizes corrected dream latents. Never feed a raw
  (un-standardized) embedding into fusion/drift/TRM, and never add normalization
  inside the TRM loss — the space is already canonical and the loss must stay
  scale-honest.
- **Parameter budget is enforced.** Fusion ≤ 5.0M, drift ≤ 1.5M, planner ≤ 2.5M, total
  < `cfg.trainable_param_budget` (9M). `tests/test_param_budget.py` and the audit will fail
  the build otherwise; do not raise a cap without the user asking.
- **Dream evidence == modality-dropout fade, one shared path.** `SlotResonanceFusion`
  weights the box + geometry tokens by `box_weight` (confidence × freshness in [0,1]);
  dream ticks pass held last-real boxes with `confidence * staleness_decay**k`, missed
  detections pass 0, and train-time `modality_dropout` fades the same weights by a
  random factor. Do NOT reintroduce binary zeroing or a separate dream flag — this
  training-inference alignment is a core design claim. Fusion's 8th token is the
  previously executed action (plan row 0) and is never faded.
- **Trust semantics.** The corrector's trust is a self-calibrating error *ratio* (EMA of
  innovation norms), and low trust HOLD-blends the plan toward the previously emitted
  plan — never multiply absolute PWM targets toward zero (that commands a real motion
  to servo mid-range).
- **Drift encoder semantics.** First forward after `reset()` stores the anchor, seeds the
  context window, and returns an exactly-zero code without stepping the GRU; hidden is
  detached between steps; runtime state (anchor, window deque, hidden) lives in plain
  attributes (never buffers/parameters). It is called on REAL ticks only; the JEPA loop
  holds its code across dream ticks. Multi-horizon lags come from `cfg.drift_horizons`
  against a `cfg.context_window`-deep memory of real-frame embeddings.
- Tests must stay CPU-only, mock-only, no network, no cv2. Use `MockTaskEncoder` /
  `MockYoloWorldPerception` / `MockTRM` (all deterministic, hash-seeded).
- Plan orientation is **rows = timesteps (5), columns = servos (7)**. When the user says
  "5x7" they may state it either way — confirm against `plan_steps`/`num_servos` in config
  rather than assuming.

## Layout

`microvla/perception/` (parser, sampler, CLIP task encoder, YOLO-World + mocks) ·
`microvla/fusion/` · `microvla/aux_state/` (drift encoder; dir is aux_state, not aux) ·
`microvla/trm/` (open slot) · `microvla/jepa/` (loop + corrector, deployment path) ·
`microvla/planner/` · `microvla/pipeline.py` (2 Hz debug path) · `train/` (BC scaffold;
episode `.npz` keys in `train/dataset.py`) · `tests/`.
