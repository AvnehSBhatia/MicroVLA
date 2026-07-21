# TRM_SPEC.md — Tiny Recursive Model handoff (v2)

This is the build spec for the TRM open slot: the world-model core that predicts the
next-tick YOLO-World frame embedding from the current fused observation and drift state.
It is **not implemented in this repo** — `microvla/trm/mock_trm.py::MockTRM` is a dumb
linear placeholder that exists only so the pipeline and JEPA loop run end-to-end before
the real model lands. Build against this document and `microvla/trm/interface.py`; the
rest of MicroVLA (`DESIGN.md`) is context, not a second source of truth for this slot.

All dimensions referenced below come from `microvla/config.py` (`MicroVLAConfig`):
`fused_rows=32`, `fused_cols=5`, `state_dim=256`, `vis_dim=512`, `tick_hz=30.0`,
`real_frame_hz=2.0`, `dream_ticks_per_real=14`. Read them from the config — do not
hardcode.

---

## 1. I/O contract and how to plug in

```python
class TRMBase(nn.Module, abc.ABC):
    def forward(self, fused: torch.Tensor, state_delta: torch.Tensor) -> torch.Tensor:
        # fused        [B, 32, 5]   float32  — SlotResonanceFusion output
        # state_delta  [B, 256]     float32  — AnchoredDriftEncoder output
        # returns       [B, 512]    float32  — predicted next-tick frame_emb
```

- `fused` rows are 32 learned slots, each a 5-wide low-rank summary of the current
  (text, frame, box, geometry) observation. On a real tick these come from grounded
  perception; on a dream tick the frame token is the *previous* corrected TRM
  prediction and the box/geometry tokens are zeroed (same fusion code path as
  train-time `modality_dropout` — see section 5).
- `state_delta` is a 256-d LayerNorm'd code from the GRU-based drift encoder, anchored
  to the first REAL frame of the episode. It is exactly zero on that anchor frame.
- Output `next_emb` is consumed by two places: `ChronoQueryPlanner(next_emb) -> plan`,
  and — at inference — the JEPA loop's own bookkeeping (it becomes the next tick's
  `pending_pred`, either checked against the next real measurement by
  `InnovationCorrector.on_measurement`, or corrected and fed back into `fused` on the
  next dream tick).

**Stateless across calls.** Episode memory lives in the drift encoder, not the TRM. The
pipeline/loop reset drift + corrector at episode start and never touch the TRM itself.
Accept any batch size `B >= 1` and preserve it; CPU-safe; constructor convention
(repo-wide) is `def __init__(self, cfg: MicroVLAConfig, ...)`.

**To plug in:** subclass `TRMBase`, implement `forward` with the exact contract above,
and pass an instance as `trm=` to either deployment surface:

```python
from microvla.trm.interface import TRMBase
from microvla.jepa.loop import JEPALoop
from microvla import MicroVLAPipeline

class MyTRM(TRMBase):
    def __init__(self, cfg):
        super().__init__()
        ...
    def forward(self, fused, state_delta):
        ...
        return next_emb

loop = JEPALoop.build_real(trm=MyTRM(cfg))          # 30 Hz deployment path
pipe = MicroVLAPipeline.build_real(trm=MyTRM(cfg))  # 2 Hz debug harness
```

No other code changes are required — everything downstream depends only on `TRMBase`.

## 2. Parameter budget: 10M

The TRM gets **10M** trainable parameters, reserved outside the 9M `fusion + drift +
planner` budget enforced by `microvla/utils/param_audit.py`. This is raised from an
earlier 7M figure — spend the extra headroom on recursion depth / a wider latent
before adding raw width; the recursion is where this architecture earns its name.
`MockTRM` is a ~0.21M single `Linear(416, 512)` and is not a size target — it is a
correctness stub only, never trained.

Full deployed-parameter picture, for context (see `microvla/utils/param_audit.py` for
the enforced side of this table):

| item | params |
|---|---|
| YOLO-World-S (frozen, incl. CLIP text tower, used once per task) | ~13M |
| **TRM (this spec)** | **10M** |
| Trainable heads (fusion ≤5.0M + drift ≤1.5M + planner ≤2.5M, hard cap 9M) | 9M |
| **Total deployed** | **~32M** |

## 3. Recommended architecture: Tiny Recursive Model

A weight-tied recursive refiner, in the spirit of the "Tiny Recursive Model" line of
work — depth from recursion, not from stacking distinct layers. Sketch (with
`d ≈ 384–512` chosen to land near 10M params):

1. **Tokenize the fused matrix.** Treat the 32 rows of `fused [B, 32, 5]` as 32
   tokens; embed each with a shared `Linear(5, d)` and add a learned 32-position
   embedding.
2. **Condition on the drift code.** Two good options (pick one, or both):
   - *Prepended token:* `Linear(256, d)` on `state_delta`, prepend as a 33rd token.
   - *FiLM:* generate per-channel `(scale, shift)` from `state_delta` and modulate
     the token stream at every recursion step. FiLM keeps the drift signal from
     washing out over K steps and is the recommended default.
3. **Recursive core (weight-tied).** A single small block — e.g. pre-LN multi-head
   self-attention over the 32(+1) tokens + a GELU MLP (hidden `2d`), or a
   token-mixing MLP block — applied **K times with shared weights** (K ≈ 4–8) to a
   latent `z_k`. Structure each step as refinement: `z_{k+1} = z_k + Block(LN(z_k),
   inputs)`, keeping the original input tokens available at every step (re-inject
   them, don't just feed the block its own output) so recursion refines rather than
   drifts.
4. **Readout head.** Pool the tokens (mean or a learned readout token) → `LayerNorm`
   → `Linear(d, 512)` producing `next_emb`.

Two training aids worth building in:

- **Deep supervision across recursion steps.** Apply the readout head (shared
  weights) to the latent after *every* recursion step and compute the §4 loss at
  each, with weights increasing toward the final step (e.g. linear ramp or
  `0.5^(K-k)`). This stabilizes weight-tied recursion, keeps early steps from
  collapsing to identity, and gives you a compute knob at inference time. This is
  independent of — and composes with — the multi-step *rollout* deep supervision in
  section 5 below (recursion steps happen *within* one tick; rollout steps happen
  *across* ticks).
- **Halting (optional).** An ACT-style halting head — a tiny per-step scalar
  `p_halt(z_k)` with a ponder-cost penalty — lets easy states exit early. Optional at
  this problem size; a fixed K is a perfectly fine v1. If you add halting, make the
  output at halt-time the deep-supervised readout of the halting step (or the
  halting-weighted mixture), and keep `forward`'s signature and output shape
  unchanged.

Any architecture is acceptable as long as §1 and §2 hold; the above is the recommended
starting point, not a mandate.

## 4. Training loss (documented only — NOT implemented anywhere in this repo)

Nothing in this repository implements TRM training; `train/losses.py`'s
`trm_loss_documentation()` only carries this spec as a returned string, referencing
this file. When you train externally:

**Target.** `y` = the *actual* YOLO-World-S `frame_emb` of the next REAL frame — never
a dream-tick's own (predicted) embedding. Targets are frozen-encoder outputs (already
detached / `no_grad` per the perception contract).

**Standardize targets.** Apply a (non-learned or frozen-affine) `LayerNorm` to `y`
before computing the loss, so the MSE term operates on a scale-stabilized target and
the two terms stay comparable across scenes. Apply the same normalization convention
to `ŷ` for the MSE term.

**Primary loss.**

```
L = λ_cos · (1 − cosine(ŷ, y)) + λ_mse · MSE(ŷ, y_ln)

λ_cos = 1.0
λ_mse = 0.5      # on LayerNorm-standardized targets
```

The cosine term carries direction (what the planner mostly consumes); the MSE term
anchors magnitude and prevents the degenerate norm-shrinking solutions cosine alone
permits.

**Optional auxiliary: in-batch InfoNCE.** Treat `(ŷ_i, y_i)` as the positive pair and
the other targets in the batch as negatives:

```
L_nce = −log( exp(sim(ŷ_i, y_i)/τ) / Σ_j exp(sim(ŷ_i, y_j)/τ) ),  sim = cosine, τ ≈ 0.1
```

Add with a small weight (e.g. 0.1–0.3). Useful when many training clips look alike and
the regression loss alone lets predictions blur toward a batch mean. Requires
reasonably diverse batches; skip it for batches drawn from a single near-static
episode.

**Collapse note (EMA / stop-grad target).** In the current design the vision encoder is
frozen, so per-tick targets are fixed and representation collapse is not a concern for
a single step. If YOLO-World (or any target-producing encoder) were ever unfrozen, do
**not** regress against the live encoder — the joint system can collapse to a constant
embedding that trivially minimizes the loss. Instead use a stop-gradient target and/or
an EMA copy of the encoder (BYOL-style, momentum ≈ 0.99–0.999) to generate `y`. This
matters more, not less, once you add the multi-step rollout of §5: a TRM that has
learned to predict a bland, low-variance "average future frame" will show artificially
low rollout loss at long horizons while being useless — watch prediction variance
across a rollout, not just its loss, as a collapse canary.

## 5. Multi-step rollout training is MANDATORY

At inference the TRM does not just do one-step prediction — it runs **up to 14-step
open-loop rollouts** between real measurements (`cfg.dream_ticks_per_real` at the
default 30/2 Hz split). Each dream tick's prediction is fed back through fusion's
dream path (`dream=True`, frame token = `InnovationCorrector.correct(prev pred)`,
box/geometry tokens zeroed) to produce the *next* tick's input. If you only ever train
the TRM on isolated one-step `(fused, state_delta) -> next_emb` pairs, you are training
a model that has never seen its own predictions come back as input — the very first
dream tick in a real rollout already puts it off its training distribution, errors
compound tick over tick, and **the corrector cannot save a bad rollout**: it only
nudges the latent by a decaying EMA of past innovations, it cannot correct a TRM that
has never learned to be stable under its own feedback.

Train with an **unrolled horizon `H`, scheduled from 1 up to 14** over the course of
training (e.g. start every batch at `H=1` for the first N steps/epochs, then grow `H`
on a schedule — linear, or +1 every fixed number of epochs — until it reaches the full
14-tick dream depth used at inference). At each unrolled step `h` (1-indexed), run the
TRM exactly as `JEPALoop.tick()` would on a dream tick: previous prediction ->
`corrector.correct` (or a straight-through/no-corrector variant if you want a stricter
lower bound — either way, exercise the fusion dream path with the fed-back
prediction) -> `fused` (dream=True) -> TRM -> `next_emb`. Compute the §4 loss `L_h` at
each step against the real frame embedding actually observed `h` ticks ahead (you need
rollout-aligned ground truth per step, not just at the end — sample training clips at
the full `tick_hz` and hold out every real-frame target along the way), and combine
with a **discount**:

```
L_rollout = sum_{h=1}^{H} 0.95^h * L_h
```

As `H` grows, this schedule exactly matches the inference feedback loop (fusion dream
path + corrector) the TRM will actually run in production — that alignment is the
entire point; training and inference must walk the identical loop. This composes with,
but is a different axis from, the intra-tick recursion-step deep supervision in §3:
recursion steps refine one tick's prediction; rollout steps chain many ticks'
predictions together.

Grow `H` on data from real episodes only (`frame_embs[t]`, `frame_embs[t+1]`, ...
`frame_embs[t+H]` from `train/dataset.py`'s episode format) so ground truth is always
real YOLO-World perception, never a previous rollout's own dream output.

## 6. Gradient flow through fusion + drift when training jointly

`TRMBase.forward` inputs are live graph tensors, so training the TRM jointly with the
upstream heads works out of the box — with these specifics, extended to the rollout
setting of §5:

- **Fusion.** Gradients flow `L_h → ŷ_h → TRM → fused → SlotResonanceFusion` (slot
  queries, cross-attention, FiLM, projections) and stop at the frozen encoder outputs
  (`command_emb`/`source_emb`/`target_emb`, `frame_emb`, box embeddings are detached
  by the perception layer). Do not `.detach()` `fused` inside the TRM. At rollout
  steps `h > 1`, gradients additionally flow *back through the previous step's TRM
  call* via the dream-mode `fusion` call that consumed `next_emb_{h-1}` as its frame
  token — this is what actually trains "be stable under your own feedback"; do not
  detach `next_emb` between rollout steps or you silently degrade §5 back into
  independent one-step training.
- **Drift encoder.** Gradients flow `L_h → TRM → state_delta →` the drift encoder's
  gate / drift-feature linear / GRUCell — but only through the **current step**:
  `AnchoredDriftEncoder` detaches its GRU hidden state between steps (BPTT is
  deliberately local, one step) and its anchor is detached at capture. Expect no
  long-horizon temporal credit assignment through the drift path's *recurrent state*
  itself, even though the rollout as a whole spans many steps — each step's
  `state_delta` only back-props one GRU cell deep.
- **The corrector is not part of the graph by default.** `InnovationCorrector` holds
  plain-Python/tensor running state (no `nn.Parameter`) and is written for inference
  bookkeeping, not backprop. Either train rollouts *without* the corrector in the loop
  (feed raw `next_emb` straight back into dream-mode fusion — the simpler, fully
  differentiable choice, and the recommended default), or reimplement its scalar
  EMA + decayed-vector-addition arithmetic inline with `requires_grad`-carrying
  tensors if you specifically want its decay behavior reflected in the training
  graph.
- **Do not backprop into raw perception.** Encoder outputs (`command_emb`,
  `source_emb`, `target_emb`, `frame_emb`, box embeddings) are produced under
  `torch.no_grad()` and detached by contract (YOLO-World-S and its CLIP text tower
  are frozen); the TRM must not try to reopen that path.
- **Optimizer hygiene.** If you optimize TRM + fusion + drift jointly, use parameter
  groups (the 9M heads generally want a smaller LR than a freshly-initialized 10M
  TRM) and remember the planner has its own behavior-cloning objective in
  `train/train_planner.py` — TRM pretraining (rollout prediction loss only) followed
  by joint fine-tuning is the intended sequencing.
- **Deep supervision interaction.** If you use per-recursion-step deep supervision
  (§3) *inside* a multi-step rollout (§5), every recursion step of every rollout step
  contributes gradient to fusion/drift through the shared input injection; that is
  fine and usually helpful, but it multiplies the effective upstream gradient by
  roughly `K × H` — rescale (normalize the summed step-losses by their weight total,
  at both the recursion and rollout levels) so upstream LRs stay meaningful as `H`
  grows over the training schedule.

---

**Deliverable checklist:** a `TRMBase` subclass, ~10M params, stateless,
`[B,32,5] × [B,256] → [B,512]`, trained externally with the §4 loss **and the
mandatory §5 multi-step rollout schedule** (`H`: 1 → 14, discount `0.95^h`), verified
by swapping it for `MockTRM` in `JEPALoop.build_real(trm=...)` /
`MicroVLAPipeline.build_real(trm=...)` and running the existing shape/pipeline/JEPA
tests.
