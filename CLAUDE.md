# MicroVLA — agent instructions

Micro vision-language-action stack (~30M params deployed) targeting a Raspberry Pi 5 with a
7-servo rig. `DESIGN.md` is the **binding architecture contract** — read it before changing
any module interface, and update it in the same change if an interface must move.
`microvla/config.py` (`MicroVLAConfig`) is the single source of truth for every dimension;
never hardcode a dim that exists there.

## Commands

```bash
.venv/bin/python -m pytest tests -q            # full suite (CPU-only, mocks, no network)
.venv/bin/python -m microvla.utils.param_audit # asserts the 9M cap + per-module caps
.venv/bin/python train/train_planner.py --epochs 2 --episodes 4   # smoke train
```

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

- **The TRM is an open slot.** `microvla/trm/` may contain only the `TRMBase` interface,
  `MockTRM`, and `TRM_SPEC.md`. Do not implement the real TRM or any TRM training code —
  it is being built externally against `TRM_SPEC.md` (contract: `(fused [B,32,5],
  state_delta [B,256]) -> [B,512]`, ~10M params). Its loss is documented in the spec and in
  `train/losses.py::trm_loss_documentation()` but must stay unimplemented here.
- **Parameter budget is enforced.** Fusion ≤ 5.0M, drift ≤ 1.5M, planner ≤ 2.5M, total
  < `cfg.trainable_param_budget` (9M). `tests/test_param_budget.py` and the audit will fail
  the build otherwise; do not raise a cap without the user asking.
- **Dream mode == modality dropout.** `SlotResonanceFusion` must keep `dream=True` and
  train-time `modality_dropout` on one shared code path (same multiplicative mask over the
  source-box/target-box/geometry tokens). This training-inference alignment is a core design
  claim; don't special-case one side.
- **Drift encoder semantics.** First forward after `reset()` stores the anchor and returns
  an exactly-zero code without stepping the GRU; hidden is detached between steps; runtime
  state lives in plain attributes (never buffers/parameters).
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
