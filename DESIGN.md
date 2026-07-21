# MicroVLA v2 — Architecture Contract

This document is the **single source of truth** for module interfaces, tensor shapes, and file
layout. Every module MUST conform to it exactly. All dims come from `microvla/config.py`
(`MicroVLAConfig`) — never hardcode a dimension that exists in the config.

v2 replaces v1 wholesale: MiniLM is DELETED (text comes from YOLO-World's internal CLIP text
tower), perception grounds an ordered SOURCE and TARGET box, the fused matrix is 32x5, the
stack runs a 30 Hz JEPA latent rollout with real perception at 2 Hz, and the trainable heads
are scaled up to a 9M budget (TRM slot reserved at 10M).

## System overview

```
                        ┌── once per task ─────────────────────────────────┐
"move can to ball" ──►  │ parse_command: source="can", target="ball"       │
                        │ YOLO-World CLIP text tower (via set_classes) ──► │
                        │ 3 ordered CLIP embs [512]: command, source, target
                        └──────────────────────────────┬───────────────────┘
                                                       │
camera 30 Hz ─┬─ every 15th tick (2 Hz) ─ REAL TICK ───▼───────────────────────────┐
              │   YOLO-World-S (frozen): frame_emb [512] (GAP of SPPF map)         │
              │     source box: emb [512] + center [2]                             │
              │     target box: emb [512] + center [2]   (per-class best box)      │
              └─ other 14 ticks ─── DREAM TICK ────────────────────────────────┐   │
                    frame token = corrected TRM prediction [512]               │   │
                    box + geometry tokens zeroed (same path as                 │   │
                    train-time modality_dropout → dream mode is trained)       ▼   ▼
  SlotResonanceFusion: 32 slots cross-attend over
      [cmd | src | tgt | frame | src-box | tgt-box | geometry] ──► fused [32, 5]
                                                                        │
  AnchoredDriftEncoder (anchor = first REAL frame, GRU accum) ──► state_delta [256]
                                                                        │
  ╔═ TRM — OPEN SLOT (~10M, built externally) ═╗                        ▼
  ║ forward(fused [B,32,5], state_delta [B,256]) -> next_emb [B,512]    ║
  ╚═════════════════════════════════════════════════════════════════════╝
                     │
                     ├──► InnovationCorrector (Kalman-lite) ──► corrected latent → next tick
                     ▼
  ChronoQueryPlanner(next_emb [512]) ──► plan [5, 7] in [-1, 1], scaled by trust τ
      rows = 5 sequential timesteps, cols = 7 servos, values = normalized PWM
```

## Parameter ledger (enforced by utils/param_audit.py + tests/test_param_budget.py)

| item | budget |
|---|---|
| YOLO-World-S (frozen, incl. its CLIP text tower used once per task) | ~13M |
| TRM (open slot, reserved) | 10M |
| Trainable heads total (HARD CAP `cfg.trainable_param_budget`) | 9M |
| — SlotResonanceFusion | ≤ 5.0M (target ~4.5M) |
| — AnchoredDriftEncoder | ≤ 1.5M (target ~0.9M) |
| — ChronoQueryPlanner | ≤ 2.5M (target ~1.6M) |
| InnovationCorrector | 0 (no learned params) |

MiniLM is gone. Total deployed ≈ 13 + 10 + ~7 ≈ 30M ≤ 32M.

## Canonical config (`microvla/config.py`) — ALREADY WRITTEN, do not modify

Key fields: `text_dim=512`, `n_text_tokens=3`, `vis_dim=512`, `state_dim=256`,
`fused_rows=32`, `fused_cols=5`, `plan_steps=5`, `num_servos=7`, `d_model=384`, `d_plan=256`,
`n_heads=8`, `n_fusion_blocks=3`, `n_planner_blocks=3`, `n_fourier=16`,
`modality_dropout=0.3`, `tick_hz=30.0`, `real_frame_hz=2.0`, `correction_beta=0.7`,
`correction_decay=0.9`, `trust_temperature=4.0`, `trainable_param_budget=9_000_000`,
properties `fps` (alias of real_frame_hz) and `dream_ticks_per_real` (=14).

## Module APIs (exact signatures)

### `microvla/perception/command_parser.py` (NEW)
```python
@dataclass(frozen=True)
class ParsedCommand:
    raw: str
    verb: str      # normalized verb phrase ("move", "pick up", ...)
    source: str    # noun phrase acted on ("can", "the red cup")
    target: str    # destination phrase; == source when the command has no destination

def parse_command(text: str) -> ParsedCommand: ...
```
Rule-based, lowercase-normalized, article-preserving. Patterns (at minimum):
`(move|put|place|push|bring|carry|slide|drag|take) X (to|onto|on|into|in|near|next to|toward|towards|at|by|behind|in front of) Y`,
`pick up X`, `grab X`, `grasp X`, `lift X`, `point (at|to) X`, `go to X`, `look at X`,
`push X (left|right|up|down|forward|back(ward)?)` (direction word becomes verb suffix,
target == source). Fallback: verb="do", source=target=full cleaned text.
Order matters: "move can to ball" → source "can", target "ball"; "move ball to can" swaps.
Pure Python, zero deps, exhaustively unit-tested.

### `microvla/perception/text_encoder.py` (REWRITTEN — MiniLM classes deleted)
```python
@dataclass
class TaskEncoding:
    command_emb: torch.Tensor  # [text_dim] float32 L2-normalized
    source_emb: torch.Tensor   # [text_dim]
    target_emb: torch.Tensor   # [text_dim]
    parsed: ParsedCommand
    def tokens(self) -> torch.Tensor: ...  # [3, text_dim] stacked (command, source, target)

class ClipTaskEncoder:
    """Harvests CLIP text embeddings from a YoloWorldPerception's model.

    encode(text): parse -> perception.model.set_classes([command, source, target]) once and
    read the internal txt_feats ([1, 3, 512], already L2-normalized) -> then leave the model's
    ACTIVE detection classes as [source, target] (or [source] when source == target) via
    perception.set_classes(...). ultralytics touched lazily, through the perception object.
    """
    def __init__(self, perception: "YoloWorldPerception"): ...
    def encode(self, text: str) -> TaskEncoding: ...

class MockTaskEncoder:
    """Deterministic (sha256-seeded per phrase) TaskEncoding; same parser, no model."""
    def __init__(self, text_dim: int = 512): ...
    def encode(self, text: str) -> TaskEncoding: ...
```

### `microvla/perception/yolo_world.py` (REWRITTEN for dual-box grounding)
```python
@dataclass
class BoxObs:
    emb: torch.Tensor     # [vis_dim]
    center: torch.Tensor  # [2] (cx, cy) in [0,1]
    xyxy: torch.Tensor    # [4] pixels (zeros if no detection)
    confidence: float     # 0.0 if fallback

@dataclass
class Perception:
    frame_emb: torch.Tensor  # [vis_dim]
    source: BoxObs
    target: BoxObs

class YoloWorldPerception:
    def __init__(self, weights: str = "yolov8s-worldv2.pt", device: str = "cpu"): ...
    def set_classes(self, classes: list[str]) -> None: ...  # ordered; role i == class i
    def perceive(self, frame_bgr: "np.ndarray") -> Perception: ...

class MockYoloWorldPerception:
    """Deterministic pseudo-perception: two distinct smoothly-moving boxes seeded from the
    frame bytes hash; same API, no model, no downloads."""
```
Implementation notes (real class, mechanics carried over from v1 where noted):
- SPPF forward hook found by module class name (as in v1); `frame_emb` = GAP, detached.
- Best box PER CLASS ID (highest confidence among that class's detections);
  `roi_align` box emb with the map's actual spatial ratio (as in v1), output 7x7, GAP.
- Missing class → fallback `BoxObs(emb=frame_emb.clone(), center=(0.5, 0.5), xyxy=zeros,
  confidence=0.0)`. One active class (source==target) → both roles share the same BoxObs.
- All under `torch.no_grad()`, detached CPU float32 outputs.

### `microvla/perception/video_stream.py` — UNCHANGED from v1 (keep the integer-counter
emit rule). Default `target_fps` now reads `DEFAULT_CONFIG.real_frame_hz`.

### `microvla/fusion/slot_fusion.py` — Slot Resonance Fusion v2
```python
class SlotResonanceFusion(nn.Module):
    def __init__(self, cfg: MicroVLAConfig): ...
    def forward(self, text_tokens, frame_emb, source_box_emb, target_box_emb,
                source_center, target_center, dream: bool = False) -> torch.Tensor:
        # text_tokens [B, 3, 512]; *_emb [B, 512]; *_center [B, 2] → fused [B, 32, 5]
```
Method (novel — slot competition over FiLM-conditioned, role-tagged modality tokens):
- 7 tokens at `d_model=384`: 3 text tokens (one shared `Linear(text_dim, d_model)`),
  frame token, source-box token, target-box token (one shared `Linear(vis_dim, d_model)`),
  geometry token = `Linear(6 * 2 * n_fourier, d_model)` over
  `concat[fourier(src_center), fourier(tgt_center), fourier(tgt_center - src_center)]`
  where `fourier(p)` = sin/cos of `p * 2^k * pi`, k in range(n_fourier) → per-point 2*2*16=64.
- Learned ROLE embedding table `[7, d_model]` added per token position (order is explicit).
- FiLM: `Linear(text_dim, 2*d_model)` from the COMMAND embedding (`text_tokens[:, 0]`)
  produces scale/shift applied to the frame, source-box, and target-box tokens.
- `dream=True` OR train-time modality_dropout (per-sample Bernoulli): zero the source-box,
  target-box, and geometry tokens — the SAME code path, so dream mode is a trained mode.
  (In dream mode the caller passes the corrected TRM latent as `frame_emb`.)
- 32 learned slot queries `[32, d_model]`; `n_fusion_blocks=3` rounds of pre-LN
  `nn.MultiheadAttention(d_model, n_heads=8, batch_first=True)` cross-attention
  (slots=queries, 7 tokens=keys/values) each + pre-LN GELU MLP (hidden `d_model*2`), residuals.
- Shared head per slot: `Linear(d_model, 64) → GELU → Linear(64, fused_cols)` → `[B, 32, 5]`.
- Params ≤ 5.0M (target ~4.5M).

### `microvla/aux_state/drift_encoder.py` — Anchored Drift Encoder v2 (scaled)
Same design as v1 (keep the v1 semantics EXACTLY: anchor stored on first forward after
reset; first call returns an exactly-zero code without stepping the GRU; hidden detached
each step; silent re-reset on batch-size change; runtime state as plain attributes), with
scaled dims: drift features `cat([emb - anchor, emb * anchor])` [B, 2*vis_dim=1024] →
`Linear(1024, state_dim=256)` → GELU → sigmoid gate `Linear(256, 256)` from the projection →
`nn.GRUCell(256, 256)` → output `LayerNorm(hidden)` [B, 256]. Params ≤ 1.5M (target ~0.9M).
`forward(frame_emb [B,512]) -> [B,256]`. NOTE: at dream ticks the pipeline feeds this the
CORRECTED latent, so it runs at 30 Hz; its anchor is always the first REAL frame.

### `microvla/trm/` — OPEN SLOT (interface + mock + spec ONLY; do NOT build the real TRM)
```python
# interface.py
class TRMBase(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, fused: torch.Tensor, state_delta: torch.Tensor) -> torch.Tensor:
        # fused [B, 32, 5], state_delta [B, 256] → next_emb [B, 512]

# mock_trm.py
class MockTRM(TRMBase):
    """PLACEHOLDER ONLY: flatten fused [B,160] cat state_delta [B,256] → Linear(416, 512).
    Replace with the real ~10M TRM."""
```
`trm/TRM_SPEC.md` — the handoff doc for the TRM builder. Must cover:
1. Exact I/O contract above + how to plug in (subclass TRMBase → pass to pipeline/loop).
2. **10M parameter budget** (raised from 7M).
3. Recommended architecture: Tiny Recursive Model — embed the 32x5 fused matrix as 32
   tokens, condition on the 256-d state delta (prepended token or FiLM), a weight-tied
   block applied recursively K≈4–8 times refining a latent, head → 512. Deep supervision
   across recursion steps; optional learned halting.
4. TRAINING LOSS (documented only, NOT implemented anywhere in this repo): target
   y = actual YOLO frame_emb at the next REAL frame;
   `L = 1.0*(1 - cosine(ŷ, y)) + 0.5*MSE(ŷ, y)` on LayerNorm-standardized targets;
   optional in-batch InfoNCE auxiliary; EMA/stop-grad target note re collapse.
5. **Multi-step rollout training is MANDATORY**: at inference the TRM runs ~14-step
   open-loop rollouts between measurements (JEPA dream ticks, predictions fed back through
   fusion's dream path). Train with unrolled horizon H (scheduled: start 1, grow to 14),
   discounted loss `sum_h 0.95^h * L_h`, matching the inference feedback loop exactly.
   Single-step-only training will compound error; the corrector cannot save a bad rollout.
6. How gradients flow back through fusion + drift when training jointly.

### `microvla/planner/chrono_planner.py` — Chrono-Query Planner v2 (scaled)
Same design as v1 (time-queried delta integration): `next_emb [B,512]` → 8 tokens of 64 →
`Linear(64, d_plan=256)` memory; 5 learned time queries + fixed sinusoidal step encoding
(registered buffer); `n_planner_blocks=3` pre-LN cross-attn blocks (8 heads, GELU MLP hidden
`d_plan*2`, residuals); per-step head `Linear(d_plan, num_servos)` predicts DELTAS;
plan = `tanh(cumsum(deltas, dim=1))` → `[B, 5, 7]` in [-1, 1]. Params ≤ 2.5M (target ~1.6M).

### `microvla/jepa/corrector.py` — InnovationCorrector (NEW, no learned params)
```python
class InnovationCorrector:
    def __init__(self, cfg: MicroVLAConfig): ...
    def reset(self) -> None: ...                 # c=0, tau=1.0, k=0
    def on_measurement(self, pred_emb: torch.Tensor, real_emb: torch.Tensor) -> None:
        # innovation e = real - pred; c ← beta*c + (1-beta)*e
        # tau ← sigmoid(trust_temperature * (cosine(pred, real) - 0.5)); k ← 0
    def correct(self, pred_emb: torch.Tensor) -> torch.Tensor:
        # returns pred + (correction_decay ** k) * c; then k += 1
    @property
    def trust(self) -> float: ...                # current tau
```
Unbatched [512] tensors at runtime. If no prediction existed yet (first real frame),
`on_measurement` is skipped by the caller. Kalman-lite complementary filter; document it.

### `microvla/jepa/loop.py` — JEPALoop (NEW)
```python
@dataclass
class TickResult:
    is_real: bool
    latent: torch.Tensor       # [512] frame emb used this tick (real or corrected)
    fused: torch.Tensor        # [32, 5]
    state_delta: torch.Tensor  # [256]
    next_emb: torch.Tensor     # [512] raw TRM prediction for the next tick
    plan: torch.Tensor         # [5, 7] in [-1,1], ALREADY scaled by trust
    trust: float
    perception: Perception | None  # only on real ticks

class JEPALoop:
    def __init__(self, cfg, task_encoder, perception, fusion, drift, trm, planner): ...
    def set_task(self, text: str) -> None:
        # encode task (sets YOLO classes), reset drift + corrector + internal latent state
    def tick(self, frame_bgr=None) -> TickResult:
        # REAL tick (frame given): perceive; if a pending prediction exists,
        #   corrector.on_measurement(pending_pred, real frame_emb); fusion grounded
        #   (dream=False, real boxes+geometry); drift(frame_emb); trm → next_emb;
        #   pending_pred = next_emb.
        # DREAM tick (None): latent = corrector.correct(pending_pred); fusion dream=True
        #   with frame_emb=latent and zeros for boxes/centers; drift(latent); trm → next_emb;
        #   pending_pred = next_emb. Raises RuntimeError if no real frame has been seen yet.
        # Every tick: plan = planner(next_emb) * corrector.trust. eval mode, torch.no_grad,
        # unsqueeze/squeeze batch dim internally.
    def run(self, frames, text: str) -> list[TickResult]:
        # frames: iterable at tick_hz (30 fps). Every int(round(tick_hz/real_frame_hz))-th
        # tick (0, 15, 30, ...) is REAL; others are dream ticks (frame ignored → None).
    @classmethod
    def build_mock(cls, cfg=None) -> "JEPALoop": ...
    @classmethod
    def build_real(cls, cfg=None, trm: TRMBase | None = None, device: str = "cpu") -> "JEPALoop": ...
        # build_real: YoloWorldPerception + ClipTaskEncoder(perception); trm defaults to
        # MockTRM with a logged warning.
```

### `microvla/pipeline.py` — MicroVLAPipeline (kept as the simple 2 Hz real-only path)
Same public API as v1 (`set_task`, `step`, `run`, `build_mock`, `build_real`) updated to the
v2 signatures/types (TaskEncoding, dual-box Perception, 32x5 fused, state 256). `step()` is
exactly a JEPA real tick without the corrector. `run()` uses VideoStreamSampler at
`cfg.real_frame_hz`. StepResult: perception, fused [32,5], state_delta [256], next_emb [512],
plan [5,7]. The JEPALoop is the deployment path; the pipeline remains for 2 Hz debugging
and as the TRM builder's minimal harness.

### `microvla/__init__.py`
Re-export: MicroVLAConfig, DEFAULT_CONFIG, MicroVLAPipeline, StepResult, JEPALoop,
TickResult, InnovationCorrector, SlotResonanceFusion, AnchoredDriftEncoder,
ChronoQueryPlanner, TRMBase, MockTRM, parse_command, ParsedCommand, TaskEncoding.

### `microvla/utils/param_audit.py`
Update to the v2 ledger (table above): heads vs 9M cap, per-module caps (5.0/1.5/2.5M),
TRM reserved 10M, YOLO-World-S ~13M frozen (CLIP text tower included, used once per task),
MiniLM row REMOVED (note: deleted in v2), MockTRM stub count. Assert total < budget AND each
module under its individual cap. Runnable via `python -m microvla.utils.param_audit`.

### `train/`
- `losses.py`: keep planner_bc_loss, smoothness_loss, total_planner_loss;
  `trm_loss_documentation()` returns the v2 spec string (32x5/256 contract, rollout
  training) referencing microvla/trm/TRM_SPEC.md — still documentation ONLY, no TRM
  training code anywhere.
- `dataset.py`: episode .npz keys v2: `frame_embs [T,512]`, `source_box_embs [T,512]`,
  `target_box_embs [T,512]`, `source_centers [T,2]`, `target_centers [T,2]`,
  `text_tokens [3,512]`, `pwm_targets [T,5,7]`; `make_synthetic_episode(T, cfg, seed)`
  generates smooth coherent fake data (boxes drifting toward each other).
- `train_planner.py`: scaffold as v1 (MockTRM slot clearly marked) updated to v2 shapes;
  add `--modality-dropout` (default cfg value) so the dream path is exercised; a few CPU
  epochs on synthetic episodes, prints losses, saves to ./checkpoints/.

### `tests/` (pytest, CPU-only, mocks only, no network, no cv2)
- `test_command_parser.py` (NEW): ≥12 patterns incl. order sensitivity ("move can to ball"
  vs "move ball to can" swap source/target), no-destination fallback, articles preserved.
- `test_shapes.py`: v2 shapes for all modules at B∈{1,4}; fusion dream=True works with
  zeroed boxes; plan in [-1,1].
- `test_pipeline.py`: mock 2 Hz pipeline end-to-end (v2 types); drift reset semantics;
  mock determinism.
- `test_jepa_loop.py` (NEW): build_mock loop; 61 frames at 30 fps → ticks 0,15,30,45,60
  real (5 real, 56 dream); dream ticks require no perception; corrector: correction decays
  over dream steps, resets counter on measurement, trust drops for orthogonal pred/real;
  dream tick before any real frame raises; all TickResult shapes; plan bounded.
- `test_param_budget.py`: v2 caps (total < 9M, per-module caps, MockTRM < 0.3M).

### Packaging / docs
- `pyproject.toml` / `requirements-full.txt`: DROP sentence-transformers everywhere.
- `README.md`: full rewrite — v2 diagram, the three novel modules + corrector, JEPA loop
  section (30 Hz story, dream-mode == modality-dropout insight, compute: YOLO only at 2 Hz,
  dream tick cost is fusion+TRM+planner ≈ 17M params → real-time CPU), v2 ledger table,
  TRM handoff (TRM_SPEC.md pointer + subclass snippet), quickstart (tests, param audit,
  train_planner smoke, real-inference snippet, JEPALoop usage), documented-not-implemented
  TRM loss summary.

## Conventions (unchanged from v1)
Python ≥ 3.10, PyTorch only, type hints, Google docstrings; lazy heavy imports (`cv2`,
`ultralytics`, `torchvision`) so `import microvla` needs only torch+numpy; every nn.Module
takes `cfg: MicroVLAConfig` first; no global seeding in library code; subpackage `__init__`s
re-export their public classes.
