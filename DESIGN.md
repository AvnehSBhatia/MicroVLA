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
                    boxes HELD from last real tick, evidence weight            │   │
                    decayed by staleness (same weighting path as               │   │
                    train-time modality_dropout evidence fade)                 ▼   ▼
  SlotResonanceFusion: 32 slots cross-attend over 8 role-tagged tokens
      [cmd | src | tgt | frame | src-box | tgt-box | geometry | last action]
      (box/geometry tokens scaled by confidence x freshness)  ──► fused [32, 5]
                                                                        │
  AnchoredDriftEncoder (anchor = first REAL frame, GRU accum,
                        steps on REAL ticks only, held during dreams) ──► state_delta [256]
                                                                        │
  ╔═ TRM — real impl at repo root TRM.py (~9.9M) ═════════════════════════════╗
  ║ forward(fused [B,32,5], state_delta [B,256], current_emb [B,512],          ║
  ║         return_box=False)                                                   ║
  ║   -> next_emb [B,512]   (RESIDUAL: current + predicted change;             ║
  ║    all embeddings in the canonical standardized space)                     ║
  ║   -> (next_emb, next_box [B,512]) when return_box  (v4: predicted next-tick║
  ║    SOURCE box emb, non-residual; the loop requests it every tick)          ║
  ╚════════════════════════════════════════════════════════════════════════════╝
                     │
                     ├──► InnovationCorrector (Kalman-lite) ──► corrected latent → next tick
                     ▼
  ChronoQueryPlanner(next_emb [512], pred_box_emb=next_box [512]) ──► raw plan [5, 7] in [-1, 1]
      emitted plan = τ·raw + (1−τ)·previous plan  (trust HOLD-blend, never →0)
      row 0 is executed this tick and fed back as fusion's action token
      rows = 5 sequential timesteps, cols = 7 servos, values = normalized PWM
```

## Parameter ledger (enforced by utils/param_audit.py + tests/test_param_budget.py)

| item | budget |
|---|---|
| YOLO-World-S detector (frozen, resident at runtime) | ~13M |
| CLIP text tower (separate ~63M model; runs ONCE per task at `set_classes`, precomputable offline — NOT resident on-device) | 0 resident |
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
    def set_role_prompts(self, source: list[str], target: list[str] | None) -> None: ...
        # per-role prompts in preference order (full phrase first, bare noun fallback)
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
- v3: `frame_emb` and every box emb are STANDARDIZED (zero mean / unit std per
  vector, `microvla/utils/embedding.py`) before leaving perception — the canonical
  embedding space every downstream consumer (fusion, drift, TRM, corrector) lives in.
- Detector class prompts are article-stripped via `strip_article` ("the red cup" ->
  "red cup"); embeddings keep the full phrases.
- Spatial grounding (Feature 1): `set_role_prompts([full_phrase, bare_noun], ...)` gives each
  role an ordered prompt list. `perceive` grounds a role to the best box of the FIRST prompt
  that detected anything — so the FULL relational phrase ("black bowl between the plate and the
  ramekin") wins when the frozen region-text head grounds it, and the bare noun ("black bowl")
  is the recall fallback. This keeps the grounded box (its center drives reaching) aligned with
  the spatial clause instead of an arbitrary same-noun box picked by raw confidence — the fix
  for LIBERO-spatial disambiguation. The JEPA loop builds role prompts from the parsed command
  (`_role_prompts`) and calls this; `set_classes` (positional role==class-id) remains for the
  legacy path and clears the role mapping. Adds NO trainable params (rides the frozen detector),
  so it improves an already-trained checkpoint at eval with no re-bake.

### `microvla/perception/video_stream.py` — UNCHANGED from v1 (keep the integer-counter
emit rule). Default `target_fps` now reads `DEFAULT_CONFIG.real_frame_hz`.

### `microvla/fusion/slot_fusion.py` — Slot Resonance Fusion v2
```python
class SlotResonanceFusion(nn.Module):
    def __init__(self, cfg: MicroVLAConfig): ...
    def forward(self, text_tokens, frame_emb, source_box_emb, target_box_emb,
                source_center, target_center,
                box_weight=None, last_action=None) -> torch.Tensor:
        # text_tokens [B, 3, 512]; *_emb [B, 512] (standardized); *_center [B, 2]
        # box_weight [B, 2] in [0,1] (confidence x freshness; None -> ones)
        # last_action [B, num_servos] in [-1,1] (None -> zeros) → fused [B, 32, 5]
```
v3 evidence weighting replaces the v2 binary dream flag: box tokens scale with their
per-role weight (geometry with the mean; weights also appended to the geometry
features), weight 0 nulls a missed detection, and the train-time `modality_dropout`
fades weights by a per-sample uniform factor — the SAME continuum dream ticks produce
with `confidence * staleness_decay**k` on held boxes. An 8th ACTION token
(`Linear(num_servos, d_model)` of the previously executed plan row) makes controlled
dynamics learnable; it is never faded.
Method (novel — slot competition over FiLM-conditioned, role-tagged modality tokens):
- 7 tokens at `d_model=384`: 3 text tokens (one shared `Linear(text_dim, d_model)`),
  frame token, source-box token, target-box token (one shared `Linear(vis_dim, d_model)`),
  geometry token = `Linear(6 * 2 * n_fourier, d_model)` over
  `concat[fourier(src_center), fourier(tgt_center), fourier(tgt_center - src_center)]`
  where `fourier(p)` = sin/cos of `p * 2^k * pi`, k in range(n_fourier) → per-point 2*2*16=64.
- Learned ROLE embedding table `[7, d_model]` added per token position (order is explicit).
- FiLM: `Linear(text_dim, 2*d_model)` from the COMMAND embedding (`text_tokens[:, 0]`)
  produces scale/shift applied to the frame, source-box, and target-box tokens.
- Evidence weighting (v3): `box_weight` scales the box tokens (geometry by the mean) and
  is appended to the geometry features; train-time modality_dropout fades the same
  weights by a per-sample uniform factor — one continuum shared with dream ticks (held
  boxes at `confidence * staleness_decay**k`; the caller passes the corrected TRM
  latent as `frame_emb`). The action token is never faded.
- 32 learned slot queries `[32, d_model]`; `n_fusion_blocks=3` rounds of pre-LN
  `nn.MultiheadAttention(d_model, n_heads=8, batch_first=True)` cross-attention
  (slots=queries, 7 tokens=keys/values) each + pre-LN GELU MLP (hidden `d_model*2`), residuals.
- Shared head per slot: `Linear(d_model, 64) → GELU → Linear(64, fused_cols)` → `[B, 32, 5]`.
- Params ≤ 5.0M (target ~4.5M).

### `microvla/aux_state/drift_encoder.py` — Anchored Drift Encoder v4 (windowed)
Semantics preserved: anchor stored on first forward after reset; first call returns an
exactly-zero code without stepping the GRU; hidden detached each step; silent re-reset
(debug-logged) on batch-size change; runtime state as plain attributes. v4 adds a
**multi-horizon context window**: a rolling deque of the last `cfg.context_window` (8)
real-frame embeddings. Per step, one drift token per reference — the anchor plus each lag
in `cfg.drift_horizons` (1, 2, 4, 8 frames ≈ 0.5–4 s at 2 Hz; lags clamp to the filled
window) — each `GELU(Linear(cat([emb-ref, emb*ref]), 256))` + a learned horizon
embedding; a single learned-query softmax attention pool reads the tokens; sigmoid gate;
`GRUCell(256, 256)` still accumulates beyond the window; output `LayerNorm(hidden)`
[B, 256]. Params ≤ 1.5M. `forward(frame_emb [B,512]) -> [B,256]`. The JEPA loop calls
this on REAL ticks only and holds the code across dreams, so the window contains only
measured evidence.

### `microvla/trm/` — OPEN SLOT (interface + mock + spec ONLY; do NOT build the real TRM)
```python
# interface.py
class TRMBase(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, fused, state_delta, current_emb) -> torch.Tensor:
        # fused [B, 32, 5], state_delta [B, 256], current_emb [B, 512]
        # → next_emb [B, 512]  (residual convention: current_emb + delta,
        #   canonical standardized space)

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
    def reset(self) -> None: ...                 # c=0, tau=1.0, k=0, err_bar=None
    def on_measurement(self, pred_emb: torch.Tensor, real_emb: torch.Tensor) -> None:
        # innovation e = real - pred; c ← beta*c + (1-beta)*e
        # SELF-CALIBRATING trust: err_bar ← EMA of ||e||;
        # tau ← exp(-0.5 * (||e||/err_bar)^2 * trust_temperature/4); k ← 0
        # (no fixed cosine threshold — real standardized frame embeddings of a
        # near-static scene are always highly correlated, so absolute-cosine
        # trust would saturate; the ratio compares the TRM to its OWN recent
        # accuracy instead)
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
    plan: torch.Tensor         # [5, 7] in [-1,1], trust-BLENDED (see tick);
                               # row 0 = action executed this tick, rows 1+ =
                               # receding horizon at 1-tick (1/30 s) spacing
    trust: float
    perception: Perception | None  # only on real ticks

class JEPALoop:
    def __init__(self, cfg, task_encoder, perception, fusion, drift, trm, planner): ...
    def set_task(self, text: str) -> None:
        # encode task (sets YOLO classes), reset drift + corrector + internal latent state
    def tick(self, frame_bgr=None) -> TickResult:
        # REAL tick (frame given): perceive (standardized embs); if a pending
        #   prediction exists, corrector.on_measurement(pending_pred, frame_emb);
        #   fusion grounded with box_weight = detection confidences and
        #   last_action = row 0 of the previously emitted plan; drift(frame_emb)
        #   (drift steps on REAL ticks ONLY); hold percept + state_delta; k=0.
        # DREAM tick (None): latent = standardize(corrector.correct(pending_pred));
        #   fusion with the HELD last-real boxes/centers and
        #   box_weight = held confidences * staleness_decay**k (k = dream ticks
        #   since the last real frame); state_delta = held value.
        #   Raises RuntimeError if no real frame has been seen yet.
        # Every tick: next_emb = trm(fused, state_delta, latent) [residual];
        #   raw = planner(next_emb); emitted plan = tau*raw + (1-tau)*previous
        #   emitted plan (HOLD-blend — low trust freezes commands, never scales
        #   absolute PWM toward the mid-range pose); plan row 0 becomes
        #   last_action for the next tick. eval mode, torch.no_grad,
        #   unsqueeze/squeeze batch dim internally.
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
- `test_shapes.py`: shapes for all modules at B∈{1,4}; fusion evidence weighting
  (fade, zero-weight nulling, action token); plan in [-1,1].
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
