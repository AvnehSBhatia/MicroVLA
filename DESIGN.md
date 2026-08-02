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
  ChronoQueryPlanner(next_emb [512], pred_box_emb=next_box [512],
                     geometry=[src_c, tgt_c, weights] [6],
                     proprio=[eef_pos, quat, gripper, valid] [10]) ──► raw plan [5, 7] in [-1, 1]
      proprio (v6): the arm's OWN state (microvla/utils/proprio.py), per tick — without it
      trajectory phase (approach/descend/lift) is unobservable from a GAP'd wrist embedding
      and MSE-BC collapses to the timid conditional mean (replay_probe: ~8x under-std,
      unchanged by v5's geometry — proprio is the missing input). Baked into npz by
      preprocess/patch_proprio.py (merge-by-episode-id, NO YOLO re-bake); zeros+valid=0
      when unavailable (Bridge, mock). npz gains OPTIONAL keys: proprio [T,10],
      eef_pos_chunk [T,plan_steps,3] (reserved for an absolute-waypoint action head).
      geometry (v5): raw grounding centers + weights handed to the planner DIRECTLY —
      for a wrist camera the target's frame position is the visual-servo error vector;
      previously it only reached the planner through fusion's 160-float bottleneck
      (trained for frame prediction), which starved control of metric geometry.
      two-stage: stage 1 predicts the 5 future 3D EEF coords (xyz waypoints = plan[...,:3]);
      stage 2 derives orientation + gripper CONDITIONED on those waypoints. Same [5,7] output,
      same split loss (pose MSE supervises the waypoints, BCE the gripper) — no loop/trainer change.
      trust is ACTION-SPACE AWARE (v5, cfg.action_space):
        "delta" (LIBERO/Bridge; zero = no motion): PROGRESSIVE brake (v5.1) — scale =
          min(1, τ/cfg.brake_trust): full magnitude while τ >= brake_trust, linear
          attenuation to a stop below it. (A flat τ·raw taxed every action by typical
          τ~0.65 even when tracking well — braking is for divergence, not a resting tax.)
        "absolute" (Pi PWM rig; zero = servo mid-range): emitted plan = τ·raw + (1−τ)·previous
          plan (HOLD-blend, never scaled toward zero)
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
- Real-tick miss hold (v5): a role whose detection MISSES on a real tick keeps its
  last-known box at `cfg.miss_decay ** age` weight (age = consecutive missed real frames)
  instead of resetting to the (0.5, 0.5)/weight-0 fallback — the wrist camera loses the
  object exactly at approach/grasp, when geometry matters most. Held per role in the
  JEPALoop (`_held_boxes`/`_miss_age`), refreshed on any hit, cleared by `set_task`.
- Symmetric action space (v5): baked `pwm_targets` are normalized so **0 <=> zero motion**
  (`preprocess/renorm_symmetric.py`; `norm_stats.json` has `q_low = -q_high`). The original
  quantile min-max mapped neutral output to the (nonzero) range midpoint — a collapsed
  policy then commanded a constant drift. Never bake a new dataset with an asymmetric
  action mapping.
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

**Input gating (v7.2).** Which memory groups the planner builds is `cfg.planner_inputs`
(subset of `ChronoQueryPlanner.INPUT_NAMES` = next_emb, current_emb, fused, state_delta,
pred_box_emb, geometry, proprio, spatial, wm_msg; at least one required). A group not
listed gets NO projection and its `forward` argument is silently IGNORED — so callers
(loop, pipeline, trainers, bench, probes) always pass everything they have and ablating an
input is a config change only. `type_emb` keeps all 11 rows at fixed indices so
checkpoints stay loadable across different `planner_inputs`. Train an ablation with
`train_batched.py --planner-drop geometry,pred_box_emb`; the choice is saved in the
checkpoint's `cfg`, so eval/bench rebuild the matching planner automatically. The evidence
for dropping one is `eval.bench --sensitivity` (on-distribution mean |Δplan| when the
input is withheld) — read `next_emb->stale` (a full-magnitude wrong prediction) alongside
`next_emb->cur` (which only zeroes the TRM's residual and so reads low by construction).

**World-model latent channel (v7.3).** `RecursiveTRM.forward_full` also returns
`latent [B, d]` — the POOLED BELIEF STATE (`out_norm(y.mean(1))`) that `next_emb`,
`next_box` and `msg` are all read from. Free: already computed. The planner
consumes it as the `wm_latent` group, chunked into `_N_MEM_TOKENS`=8 tokens
through one shared `Linear(wm_latent_dim/8, d_plan)` — 33K params, and 8 tokens
so it can compete with `fused`'s 32 for attention rather than arriving as a
single token. `cfg.wm_latent_dim` (1024) must match the TRM's `d`.

Why: measured on the +19.8% stage-A checkpoint (paper.md §4h), that pooled state
is vision-rich — zeroing `fused` destroys 89% of the TRM's predicted residual and
box evidence drives 38–41% of it — while `msg`, its 32-wide readout, collapsed to
**92% a fixed vector** (constant norm 3.32 vs varying 0.268) at an effective rank
of 6/32. A near-constant input is absorbable into the consumer's bias, which is
why the planner's measured sensitivity to `msg` was 0.0006. The bottleneck was
the channel, not the planner.

`latent` is OPTIONAL in the TRM readout contract: callers use `wm.get("latent")`
and the planner ignores `None`, so the zero-parameter foils in `eval/baselines.py`
— which genuinely have no belief state — remain valid `TRMBase`
implementations.

**Waypoint-absolute actuation (v7.2, opt-in `cfg.waypoint_action`).** The magnitude lever.
Every previous fix for the collapse (`std_ratio` 0.12 → 0.37, healthy ~1.0) attacked the
INPUTS of the action regression; this attacks its OUTPUT. Extra head
`wp_disp_head: Linear(d_plan, 3)` (771 params) on the same waypoint-conditioned features
`h`, same `tanh(cumsum(·))` structure → `[B, 5, 3]` in [-1, 1] = metric EEF displacement
from the CURRENT position in units of `cfg.waypoint_range` (0.15 m). Returned by
`planner(..., return_wp=True)` and carried on `TickResult.waypoints`; `None` when the head
is off, so every existing caller is unchanged.
- **Supervision**: `microvla/utils/waypoint.py::waypoint_targets` — plan row `k` is
  supervised against `(eef_pos_chunk[k+1] - eef_pos_chunk[0]) / waypoint_range`. The bake
  carries `plan_steps` rows, so the LAST row has no target and is masked (4/5 rows, always
  including row 0 — the only executed one). `train/losses.py::waypoint_loss` masks BOTH
  that row and any sample whose proprio validity flag is 0 (a zero-filled episode would
  otherwise teach "the arm never moves" — the exact collapse this head exists to fix).
  Train with `train_batched.py --waypoint-weight 1.0`.
- **Actuation** (`WaypointActuator`, eval-side because it needs raw action units): the
  absolute target `eef_measured + disp[horizon]` is refreshed on REAL ticks and HELD across
  the dream ticks between, and the command is `gain_scale · (target − eef_measured) / gain`
  clipped to ±1. Holding it is the point: the command tracks the REMAINING positional
  error, so a timid prediction makes the arm arrive late rather than never. It replaces
  only the translation dims; orientation and the gripper still come from the plan, and the
  delta-mode trust brake still scales the emitted command.
- **`gain`** (metres of EEF travel per unit raw action per control step) is fitted from
  data by `preprocess/fit_waypoint_gain.py` → `waypoint_stats.json`, and must be PAIRED
  WITH ITS CHECKPOINT exactly like `norm_stats.json` (a gain fitted under a different
  action normalization is meaningless). `eval.libero_eval --waypoint-stats <path>` turns
  actuation on; without the file, or on a checkpoint with no head, or on a step with no
  proprio, the plan drives all seven dims as before.
- **Measurement**: `eval.bench` reports `wp_std_ratio` / `wp_mae_mm` — the head's own
  fidelity in metres, needing neither normalizer nor gain. The lever only pays if
  `wp_std_ratio` beats the action `std_ratio` on the same checkpoint.

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
        # Every tick: (next_emb, next_box) = trm(..., return_box=True) [residual];
        #   raw = planner(next_emb, current_emb, state_delta, fused,
        #   pred_box_emb=next_box, geometry=[src_c, tgt_c, weights]);
        #   trust is ACTION-SPACE AWARE (v5): "delta" -> emitted = tau*raw
        #   (low trust BRAKES toward zero motion; holding a delta is momentum);
        #   "absolute" -> emitted = tau*raw + (1-tau)*previous plan (HOLD-blend,
        #   never scaled toward the mid-range pose). Gripper column is always a
        #   hard +/-1 (sign of raw). Plan row 0 becomes last_action for the next
        #   tick. eval mode, torch.no_grad, unsqueeze/squeeze batch internally.
    def run(self, frames, text: str) -> list[TickResult]:
        # frames: iterable at tick_hz (30 fps). Every int(round(tick_hz/real_frame_hz))-th
        # tick (0, 15, 30, ...) is REAL; others are dream ticks (frame ignored → None).
    @classmethod
    def build_mock(cls, cfg=None) -> "JEPALoop": ...
    @classmethod
    def build_real(cls, cfg=None, trm: TRMBase | None = None, device: str = "cpu") -> "JEPALoop": ...
        # build_real: YoloWorldPerception + ClipTaskEncoder(perception); trm defaults to
        # MockTRM with a logged warning.
    @property
    def device(self) -> torch.device: ...        # read from fusion, fresh each tick
```
**Device placement (v7.2).** `JEPALoop.device` is read from `fusion` on every tick (not
cached), so `loop.fusion.to(...)` after construction is honoured. Perception is deliberately
NOT bound to it: the detector may sit on a GPU while the heads sit anywhere, and every
tensor crossing the boundary (`Perception`, the TQSA feature map, proprio, box weights,
geometry, text tokens) is moved at that boundary. `Tensor.to` is a no-op when the device
already matches, so the all-CPU default costs nothing. This matters because the heads run at
`tick_hz` and the detector at `real_frame_hz` — on the 15:1 schedule the d=1024 TRM, not the
detector, dominates eval wall-clock, and `eval.libero_eval --heads-device` is how you move
it. `InnovationCorrector`'s accumulator adopts the embeddings' device (plain runtime state,
never a buffer — same rule as the drift encoder's).

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

## v7 plan — trainable spatial perception + teacher distillation (BINDING once built)

Motivation (evidence chain in paper.md "Action-interface diagnosis"): the policy's
~8x action-magnitude collapse survived symmetric actions, direct geometry, and
dream-consistent training — the observation itself is the ceiling. GAP destroys
spatial structure; frozen detector features were trained for detection, not
manipulation; 50-demo human teleop is a noisy BC target. v7 attacks all three.

1. **One re-bake pass per suite** (download -> convert -> delete, BudgetGuard):
   npz gains `wrist_frames [T, 128, 128, 3] uint8` (compressed; ~1-1.5 GB across
   LIBERO — inside the 10 GB cap). Frames make perception TRAINABLE forever after
   (no more re-downloads). Converter output must be re-normalized symmetric
   (`preprocess/renorm_symmetric.py`, idempotent) until the converter bakes
   symmetric stats natively. `wrist_frames` is NOT in EPISODE_KEYS/OPTIONAL_KEYS
   (too big to zero-fill); the v7 trainer loads it explicitly.
2. **Text-Queried Spatial Adapter (TQSA)** — `microvla/perception/spatial_adapter.py`,
   trainable (~0.3M), on the FROZEN YOLO-World backbone's hooked SPPF map:
   1x1 conv 512->128, per-role text projections (command/source/target CLIP embs),
   attention maps = softmax_HW(text_j . feat_hw / sqrt(128)) — the task-conditioned
   "CLIP attention maps" — plus attention-pooled role vectors [3, 128] and a 4x4
   spatial-token grid [16, 128]. Fusion consumes the pooled role vectors (one new
   token); the planner cross-attends the spatial tokens + downsampled heatmaps.
   The frozen detector KEEPS doing boxes (open-vocab grounding, miss-hold, role
   prompts — unchanged). Full-backbone fine-tune deliberately NOT default (6k
   episodes would destroy open-vocab grounding); an opt-in partial-unfreeze flag
   may be added for the ablation.
3. **Teacher distillation as the primary Stage-B signal**: converter's existing
   `--teacher tinyvla` path relabels action chunks at bake time (teacher sees the
   raw frames + instruction; cache per episode id). Student loss = distill MSE/BCE
   on teacher chunks + demo-BC auxiliary. TinyVLA first (small footprint, scaffold
   exists); pi0-family LIBERO-finetuned checkpoints are the stretch teacher
   (stronger, but ~7 GB transient weights against the 10 GB cap).
4. **Waypoint-absolute action head** (uses baked `eef_pos_chunk` + `proprio`):
   stage 1 predicts DISPLACEMENTS to the next `plan_steps` EEF waypoints from the
   MEASURED current EEF pose at every replan — policy errors cannot integrate
   (each tick re-anchors on ground truth), and position targets are far better
   conditioned for MSE than per-step deltas. Execution: raw action = per-dim
   fitted gain x displacement row 0 (gain fitted from demo (action, Δeef) pairs
   at patch/bake time, stored beside norm stats).
5. **Eval throughput** (SHIPPED): `--workers N` process-sharded tasks + policy-
   camera-only rendering; N workers ~= N x wall-clock on the CPU-bound osmesa path.

Build order: P1 re-bake (frames [+ teacher] + proprio + symmetric) -> P2 TQSA +
planner/fusion/trainer wiring -> P3 stage A (unchanged objective) + stage B
(distill + waypoint) -> P4 probe -> parallel eval. The world-model contract
(TRM v4, JEPA loop v5.1) is unchanged by v7.

## v8 plan — relational reasoning after the TRM, HRM long-horizon backbone (BINDING once built)

Locked 2026-07-26 on explicit request. v8 replaces three of five trainable
modules. Every v7 checkpoint is incompatible, including
`full_stageA_wrist_v72.pt` — the +19.8% `wm_margin` result must be re-earned by
a full retrain before any v8 number exists.

### Motivating evidence

Three measurements drive this, all from `paper.md` §4m:

1. **Closed-loop collapse is directional, not magnitudinal.** Per-axis `|cmd|`
   on the overnight run was x 0.1186, y 0.8550, z 0.2420 — one axis at 7.2x
   another, sustained over 3000 steps at 0% clipping. Which axis dominates
   differs per checkpoint (an earlier run: x 0.5682, y 0.2339, z 0.4174). The
   policy emits a near-constant direction that is a per-run artifact.
2. **On-distribution variance is healthy** (`wp_std_ratio` 0.75–0.94) while
   closed-loop behaviour is constant, which is exposure bias, not underfitting.
3. **Nothing in v7 reasons about object-object relations**, yet every task in
   the corpus is relational ("put the soup IN the basket").

### Ordering change

v7 ran fusion -> TRM. v8 runs **TRM -> relational**: the TRM does temporal
prediction (the one component with a positive result), and relational reasoning
then operates on the predicted latent — the same state the planner is
conditioned on, rather than a separate pre-TRM summary.

```
perception (frozen YOLO-World-S)   DATA RICH: K=cfg.max_objects proposals at
   |                               full vis_dim, no [32,5] bottleneck
   |  frame_emb [B,512] + obj_emb [B,K,512] + obj_center [B,K,2] + obj_weight [B,K]
   v
HRMBackbone      (replaces AnchoredDriftEncoder)
   |  slow module steps on REAL ticks (real_frame_hz); fast module every tick
   |  -> HRMState(state [B,hrm_dim], gains [B,hrm_gain_dim])
   v
RecursiveTRM     (unchanged contract, residual convention preserved)
   |  -> next_emb [B,512]
   v
RelationalHead   (replaces SlotResonanceFusion)
   |  cross-attn(obj tokens x next_emb x text), obj_weight fades PROJECTED content
   |  -> [B, rel_tokens, rel_dim]
   v
ChronoQueryPlanner -> plan [B,5,7] + waypoint [B,5,3]
```

### Why an HRM specifically

An HRM's two coupled timescales are not an arbitrary import: the hierarchy
already exists in the deployment loop. The slow module steps only on real
perception ticks (2 Hz), the fast module every tick (30 Hz), and the fast module
converges toward a local equilibrium between slow updates — which is exactly the
dream-tick regime. It subsumes three jobs v7 did separately or by hand: drift
encoding, the hand-fitted per-axis proportional gains of
`preprocess/fit_waypoint_gain.py` (x 0.01085, y 0.01306, z 0.01180 — now learned
outputs, "learned PID"), and long-horizon reasoning over `cfg.context_window`.

### Exact signatures

```python
# microvla/relational/relational_head.py
class RelationalHead(nn.Module):
    def __init__(self, cfg: MicroVLAConfig) -> None: ...
    def forward(
        self,
        next_emb: torch.Tensor,       # [B, vis_dim]   TRM's predicted latent
        obj_emb: torch.Tensor,        # [B, K, vis_dim]  K = cfg.max_objects
        obj_center: torch.Tensor,     # [B, K, 2]
        obj_weight: torch.Tensor,     # [B, K]  confidence x freshness, [0,1]
        text_tokens: torch.Tensor,    # [B, 3, text_dim]
        last_action: Optional[torch.Tensor] = None,   # [B, num_servos]
    ) -> torch.Tensor:                # [B, rel_tokens, rel_dim]

# microvla/hrm/hrm_backbone.py
@dataclass
class HRMState:
    state: torch.Tensor               # [B, hrm_dim]
    gains: torch.Tensor               # [B, hrm_gain_dim], strictly positive

class HRMBackbone(nn.Module):
    def __init__(self, cfg: MicroVLAConfig) -> None: ...
    def reset(self) -> None: ...
    def forward(self, frame_emb: torch.Tensor, is_real: bool = True,
                eef: Optional[torch.Tensor] = None) -> HRMState: ...

# microvla/perception/text_region.py
class TextRegionExtractor:            # ZERO trainable params
    """Top-K class-agnostic proposals with embeddings in YOLO-World's TEXT
    space, via a hook on WorldDetect.cv4 (verified present: WorldDetect
    children == ['cv2', 'cv3', 'dfl', 'cv4'])."""
```

### Carried over unchanged (do not re-litigate)

* **Graded evidence fade, one shared path.** `obj_weight` multiplies PROJECTED
  object content before any type/positional embedding. Dream ticks pass held
  boxes at `confidence * staleness_decay**k`; misses pass 0.0; train-time
  `modality_dropout` fades the same weights. No binary zeroing, no dream flag.
  The last-action token is never faded.
* **TRM residual convention** (`return current_emb + delta`) and statelessness;
  the context window stays caller-owned.
* **HRM runtime-state semantics inherited from the drift encoder**: first
  forward after `reset()` returns an exactly-zero code without stepping; hidden
  detached between steps; anchor/window/hidden in plain attributes, never
  buffers or parameters, so a checkpoint never carries episode state.
* **Canonical embedding space** — standardize at the perception boundary, never
  normalize inside a module or a loss.

### Parameter ledger

| module | v7 | v8 target | cap |
|---|---|---|---|
| fusion -> relational | 4,460,165 | ~2.4M | 5,000,000 (inherited) |
| drift -> HRM | 724,993 | ~2.5M | 3,000,000 (raised from 1,500,000) |
| planner | 1,803,527 | 1,803,527 | 2,500,000 |
| **total** | 6,988,685 | **~6.7M** | 9,000,000 (unchanged) |

The HRM cap rise is the only budget change, granted because the module absorbs
work v7 did in three places. The joint `cfg.trainable_param_budget` is untouched
and still binds.

---

## v9 — AS-BUILT: the architecture that completed the task (2026-08-01, BINDING)

This section documents, layer by layer, the exact deployed stack behind the
first completed LIBERO pick-and-places (cream cheese 0.20, alphabet soup
0.75, assisted/calibrated track — see `paper/paper.md` §5r–§5t and
`paper/MANUSCRIPT.md`). Checkpoint of record:
`checkpoints/full_stageB_rec_fix.pt` (16.584M trainable across 211 tensors;
frozen YOLO-World-S 13M on top ⇒ ~30M deployed). Live-path annotations come
from the weight-forensics ledger (`paper/forensics_ledger.md`, 554 entries).

### 0. Dataflow at a glance (dims are the checkpoint's actual config)

```
command text ─parse→ (source, target) phrases
   └─CLIP text tower (frozen, once/task)→ text_tokens [3,512]  (cmd, src, tgt)

frame BGR (2 Hz real ticks) ─YOLO-World-S (frozen)─┐
   frame_emb [512]  (SPPF hook, standardized)      │
   K=8 proposals: emb [8,512], center [8,2],       │
     confidence → obj_weight [8]                   │
   role boxes: source/target emb+center+conf       │
   spatial_grid [16,512] (4×4 ROI features)        │
                                                   ▼
 ┌───────────── per-tick trainable stack (30 Hz) ────────────────┐
 │ EvidenceEncoder (fusion slot, 0.116M)                          │
 │   [frame|objs|text|last_action] → fused [32,5]                 │
 │ HRMBackbone (drift slot, 2.110M)                               │
 │   two-rate recurrent state → state_delta [256] (+gain code)    │
 │ RecursiveTRM (9.969M, d=1024)                                  │
 │   (fused, state_delta, current_emb[512], context[≤8,512])      │
 │   → next_emb [512] (residual), next_box [512], msg [32],       │
 │     latent [1024]                                              │
 │ TQSA (0.132M): text × spatial_grid → spatial token [128]       │
 │ RelationalHead (2.355M, d=384)                                 │
 │   cross-attn(12 queries × [objs, text, next_emb, action])      │
 │   → rel tokens [12,384]      ◄── 97% of plan ablation impact   │
 │ ChronoQueryPlanner (1.901M, d_plan=256)                        │
 │   5 time-queries cross-attend 3 blocks over the token memory   │
 │   → plan [5,7] (tanh), grip logits [5], waypoints [5,3]        │
 └────────────────────────────────────────────────────────────────┘
                                                   ▼
 JEPALoop 30 Hz: real tick every 15th (perception_period at eval: 2);
 dream ticks feed corrected next_emb back through the SAME evidence path
 with box confidences × staleness_decay^k (0.9^k). InnovationCorrector
 (no params) rescales trust τ; "delta" action space brakes plan by
 min(1, τ/0.5).
                                                   ▼
 CALIBRATED CONTROL LAYER (eval/ibvs_phase.py, zero learned params) —
 the assisted track that produced the successes; see §5 below.
```

### 1. Perception (frozen, 13M)

* YOLO-World-S with its own CLIP text tower. `ClipTaskEncoder` harvests
  (command, source, target) text embeddings ONCE per task — the stack
  contains no other language model.
* SPPF forward hook → frame embedding; ROIAlign over detector proposals →
  per-object embeddings; per-role (source/target) binding by text-region
  matching with `role_disjoint_iou 0.1`, `source_max_area 0.12`,
  `det_conf 0.02` at eval.
* EVERY visual embedding is standardized (zero mean/unit std per vector,
  `microvla/utils/embedding.py::standardize`) at the perception boundary —
  the canonical-space rule. Never re-normalize downstream; the spec loss
  depends on it (§4).

### 2. Trainable modules, layer by layer

#### 2.1 EvidenceEncoder — `microvla/relational/evidence.py` (fusion slot, 0.116M)
```
obj_proj    Linear(512+2 → 96)     # per-object [emb|center], conf-weighted mean-pool
frame_proj  Linear(512 → 96)
text_proj   Linear(512 → 96)
action_proj Linear(7 → 96)         # previously EXECUTED action (plan row 0), never faded
assemble    Linear(4·96 → 160) → reshape [32, 5] = fused
```
Evidence weighting: object tokens scale by `box_weight = confidence ×
freshness ∈ [0,1]`; dream ticks pass held boxes decayed by 0.9^k; train-time
modality dropout fades the SAME weights (no dream flag anywhere — the core
alignment rule). Feeds the TRM's unchanged [32,5] port.

#### 2.2 HRMBackbone — `microvla/hrm/hrm_backbone.py` (drift slot, 2.110M, d=256)
```
SLOW module (steps on REAL ticks only):
  drift_proj   Linear(drift_feats → 256)   # multi-horizon lags (1,2,4,8) vs
  horizon_emb  Param[4, 256]               #   a context_window=8 memory
  ctx_attn     MHA(256, heads, batch_first) + q/kv LayerNorms
  blocks       N× [LN → Linear(256→512) → GELU → Linear(512→256)] residual
  rate         Linear(256→256)             # gated two-rate state update
FAST module (every 30 Hz tick):
  fast_in      Linear(2·512 → 256)         # [current_emb | predicted emb]
  eef_proj     Linear(eef_feats → 256)     # proprio stream
  slow_to_fast / fast_to_slow  Linear(256→256) cross-rate exchange
  out_norm     LayerNorm → state_delta [256]
READOUT (30 Hz path):
  gain_head    Linear(256 → 3), ZERO-INITIALIZED by design;
               modulation = GAIN_LOG_RANGE · tanh(gain_head(code))
```
⚠ Forensics: `gain_head.weight` is still EXACTLY its zero init after
training (ledger F-003) — the learned per-axis action-gain mechanism never
received gradient. This is the leading suspect for the §4p 2–4× action
magnitude shrink and the first thing the next training run must fix.
Runtime state (anchor, window deque, hidden) lives in plain attributes;
first forward after `reset()` returns an exact-zero code (anchor tick).

**GRAM experiment (opt-in, `cfg.gram_hrm` / `cfg.gram_planner`).** Baek et al.
2026 stochastic residual guidance — **on the HRM slow update and on planner
features before the waypoint/orient/grip/wp_disp heads, never on the TRM.**
Shared primitive: `microvla/utils/gram.py::StochasticGuidance`. Off by
default (zero-init → deterministic identity until trained). Inference width:
`cfg.gram_n_samples` parallel trajectories, mean pose + majority grip.

#### 2.3 RecursiveTRM — root `TRM.py` (9.969M, d=1024, T=3, n_inner=6)
```
embed      Linear(5+3·16 → 1024)   # per-slot: [fused row | current chunk |
                                   #   fast-history chunk | slow-history chunk]
ctx_decay  Param[2, 8]             # learned fast/slow softmax reads over the
                                   #   caller-owned context window [≤8, 512]
pos        Param[32, 1024]
film       Linear(256 → 2048)      # state_delta → per-slot (scale, shift)
net        TinyNet (WEIGHT-TIED, called T·(n_inner+1) times per pass):
             norm1 LN(1024); token_mix Linear(32→32) over slots;
             norm2 LN(1024); chan_mlp Linear(1024→4096) GELU Linear(4096→1024)
y_init, z_init  Param[1024]        # two-latent (y, z) recursion scheme
out_norm   LN(1024); head Linear(1024→512)        # pooled → residual Δ
box_head   Linear(1024→256) GELU Linear(256→512)  # next SOURCE box emb
msg_head   Linear(1024→32)         # action-shaped channel (stage-B unfrozen)
```
Contract (v3): `forward(fused, state_delta, current_emb, context) =
current_emb + Δ` — the RESIDUAL convention. Inference runs ONE refinement
pass (`n_sup_infer=1`, ~19 ms CPU); deep supervision (n_sup=3) is
training-only via `refine_forward`. `forward_full` additionally returns
`next_box`, `msg`, and the pooled `latent` (exported because `msg` collapsed
to a near-constant, eff-rank 6/32 — §4h).
Forensics: dreaming is intrinsically stable — closed 30-step rollouts orbit
a shared attractor (contraction 0.42×), Jacobian σ₁ = 1.000 (ledger D-001..3).

#### 2.4 TQSA — text-queried spatial attention (0.132M)
```
t_proj  Linear(512 → 128)   # text queries over the 16-cell spatial grid
→ spatial token [128] + an 8×8 heat readout
```
Forensics: `t_proj` effective rank 28.5/128 — the corpus's task phrases
compressed language conditioning to a task-ID lookup (ledger F-012).

#### 2.5 RelationalHead — `microvla/relational/relational_head.py` (2.355M, d=384)
```
visual_proj Linear(512→384)  text_proj Linear(512→384)
action_proj Linear(7→384)    geom_proj Linear(n_fourier+1 → 384)
rel_bias    Linear(n_fourier → 8)   # pairwise geometry bias per head
type_emb    Param[n_types, 384]     queries Param[12, 384]
2 × [ LN → MHA(384, 8 heads) → LN → Linear(384→hidden) GELU → Linear(→384) ]
out_norm    LN(384) → 12 relational tokens
```
Runs AFTER the TRM (the v8 ordering bet): object tokens (conf-weighted,
staleness-faded on dream ticks), text tokens, the predicted latent, and the
last action attend into 12 query tokens. Forensics: these 12 tokens carry
**97% of the planner's ablation-measured input dependence** — this head IS
the perception→action interface (ledger D-004).

#### 2.6 ChronoQueryPlanner — (1.901M, d_plan=256, 3 blocks, 8 heads)
```
Token memory (type_emb Param[14,256] tags each source):
  mem_proj(next_emb) · cur_proj(current_emb) · state_proj(state_delta)
  · proprio_proj(10→256) · spat_proj(128→256) · heat_proj(64→256)
  · msg_proj(32→256) · wm_latent_proj(chunked 1024) · wm_delta_proj
  · rel_proj(384→256) ×12 · [fused_proj/box_proj/geom_proj when configured]
time_queries Param[5, 256]  (zero-init; one per plan row)
3 × _CrossAttentionBlock: LN(q) → MHA(256, 8) vs LN(memory) → LN →
                          Linear(256→512) GELU Linear(512→256), residual
final_norm LN(256)
heads: pose (tanh-bounded cumulative updates → plan [5,7] rows=timesteps),
       grip_head Linear(256→1) per row (BCE logit, hard ±1 at deploy),
       waypoint_head Linear(256→3·5) metric EEF displacements,
       wp_proj feeds waypoints back as a token
```
Planner inputs in the checkpoint: `(next_emb, current_emb, state_delta,
proprio, spatial, wm_msg, wm_latent, wm_delta, relational)`. Forensics:
ablation impact concentrates in `relational` (0.3615) ≫ `current_emb`
(0.0120) ≫ everything else ≤ 0.0024 — candidates for pruning at deploy.

### 3. Runtime: JEPALoop + corrector (no learned params)

* 30 Hz tick; every `perception_period`-th is REAL (eval used 2; the design
  point is 15). Real tick: perceive → drift slow-step → TRM `forward_full`
  → planner. Dream tick: corrected prediction re-enters the SAME evidence
  path with staleness-faded boxes; drift code held; context window
  (caller-owned, ≤8 latents) appended each real tick.
* InnovationCorrector: trust τ is a self-calibrating EMA ratio of innovation
  norms. `action_space="delta"` ⇒ plan × min(1, τ/brake_trust=0.5)
  (progressive BRAKE — holding a stale delta is momentum); `"absolute"` ⇒
  hold-blend toward the previous plan, never scale to zero. Eval of record
  ran `--no-brake`.

### 4. Losses (exact forms)

Stage A (world model: fusion+drift+TRM, deployment-exact 15-tick rollouts,
scheduled horizon H 1→6, discount 0.95^h):
```
L_A = Σ_h 0.95^h · [ 1 − cos(ŷ_h, y_h) + 0.5·MSE(ŷ_h, y_h) ]   (spec_loss)
    (+ box-head MSE on next source box emb, same standardized space)
```
RAW vectors — the space is already canonical; normalizing inside the loss
would forgive the scale errors that poison the dream feedback loop.

Stage B (planner BC through the frozen world model; TRM core frozen,
`msg_head` deliberately unfrozen so the policy gradient shapes the message):
```
L_B = wMSE(pose rows, pwm_targets)          # row0_weight=2 (executed row),
                                            # pre-grasp step weights (phase.py)
    + grip_weight · BCE(grip_logits, target_grip>0)    # bimodal gripper
    + smooth_weight · ‖Δ²(pose rows)‖²      # second-difference smoothness
    + variance_weight · magnitude term      # anti-conditional-mean-shrink
    + waypoint MSE (row/validity masked)    # when the waypoint head trains
    (+ optional IBVS-shaped centering_loss / depth_loss on grasp windows,
       modality_consistency_loss on the dream path — measured negative for
       the centering family on this task, see paper §5q)
```
Recipe of record (rec_fix): recovery-noise 0.01, variance 0.1,
action-token-sampling 0.5, v8 arch, TQSA on.

### 5. The calibrated control layer (the piece that closed the task)

`eval/ibvs_phase.py::PhasedIBVS` — zero learned parameters, owns the action
under `--ibvs-phase`. Phases:
```
servo_src → (gate: z < gate_z, err inside band, BIND VERIFIED)
  → align   (proprio-only P-servo to eef + grasp_offset, probe dx per retry,
             fly-over above approach_z, optional yaw probe)
  → grasp   (descend to close_z / z-stall contact; press while closing;
             jaw check = ABS-mean of mirrored finger joints ≥ 0.2)
  → lift → transport (proprio-only to place_at) → release (lower to drop_z,
             open) → done;  air-close → rise (retry_rise ticks) → align
```
Calibrated constants — ALL offline, from logged runs / demo statistics,
never from sim state at runtime:

| constant | value | provenance |
|---|---|---|
| camera→gripper lever arm | (+0.080, −0.050) m | 231 at-gate episodes |
| probe schedule | dx ∈ {0,±2,±4,+6} cm (× optional 90° yaw) | at-gate variance |
| close_z / gate_z / approach_z | cream .01/.06/0 · soup .045/.10/.12 · dressing .114/.17/.20 | demo close heights + geometry |
| basket point | (−0.005…−0.010, 0.255…0.260) | 50 demo end-states/task |
| drop_z | 0.18–0.30 | basket rim |
| gate bind verify | CLIP margin: src-cosine > tgt-cosine AND no better source proposal >0.10 away | defect-29 sibling analysis |

Two defects this layer's telemetry exposed (both now fixed, both class-2
"agreement on a wrong convention"): the aim-invariant lever arm (§5r) and
defect 29 — `mean(gripper_qpos)` over MIRRORED (+q,−q) panda fingers ≡ 0 in
every state, which had discarded every physically successful grasp in
project history one tick before lift (fixed with `abs()` in `PhasedIBVS`
and `GraspToolController`).

### 6. Status of the claims

The learned stack (perception → … → planner) is the measured, budgeted,
novel architecture; the calibrated layer is the assisted track that proves
the frozen features + three constants suffice for the full task (cream 0.20,
soup 0.75, n=8–10, honest per-trial tables in `paper/paper.md`). The
distillation path (record the machine's successful rollouts, stage-B BC on
them, fix the dead `gain_head` gradient first) is the sanctioned route to
converting these numbers into UNAIDED policy success. Do not blur the two
tracks in any claim.
