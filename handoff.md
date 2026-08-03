# Handoff — first successes: calibrated pick-and-place (2026-08-01, evening)

**Audience:** next agent/human picking up MicroVLA closed-loop work.
**Contract:** `DESIGN.md` + `microvla/config.py`. Narrative evidence:
`paper/paper.md` §5r–§5t (note: paper.md MOVED to `paper/`). Submission
draft: `paper/MANUSCRIPT.md`. Weight forensics: `paper/forensics_*.md`,
figures in `paper/visuals/`.

---

## 1. One-paragraph status

**The zero is broken.** `handeye_v4` (cream cheese, task 1, wrist camera,
rec_fix checkpoint + calibrated `PhasedIBVS`): **mean_success 0.200 (n=10)**
— the first completed pick-and-places in project history (previous record:
0.000 over 347 real evals). Two more trials grasped and ran out of steps at
max-steps 400 (raised to 600 since). The unlock was three measured fixes in
one session: (1) a constant ~8.9 cm camera→gripper lever arm that no aim
sweep could touch (grasp gate fires on z-crossing, not image convergence) —
fixed by a proprio-only `align` phase to `eef + (+0.08, −0.05)` with ±6 cm
probe retries; (2) **defect 29**: the held-object jaw check took the SIGNED
mean of the panda's mirrored finger joints (+q, −q) ≡ 0 in every state —
every good grasp ever made was discarded one tick before lift; one `abs()`
fixed it in BOTH machines (`eval/ibvs_phase.py`, `microvla/tools/
grasp_tools.py`); (3) the place leg is proprio-only to the demo-calibrated
basket point (fixed across all 50 demos per task, std < 2.5 cm). Honesty:
this is the ASSISTED track (state machine owns actions; constants calibrated
offline from logged runs/demos; no sim ground truth at runtime).

## 2. Pod / ops (unchanged basics)

| item | value |
|---|---|
| SSH | `ssh -p 15266 -i ~/.ssh/id_ed25519 root@213.173.107.139` |
| Repo on pod | `/root/MicroVLA` (rev 02f38d5 + rsynced eval/ + tools/ changes — NOT committed) |
| Best ckpt | `checkpoints/full_stageB_rec_fix.pt` (also pulled LOCAL for forensics) |
| Norm stats | `data/libero_object_grid/norm_stats.json` (also local now) |
| pkill warning | never `pkill -f <pattern>` where your own ssh cmdline contains the pattern — use `ps aux \| grep "[x]yz"` |
| argparse warning | `--ibvs-place-at=-0.005,0.257` needs the `=` form (leading dash) |
| bash-chain warning | `export BASE=...` before `nohup bash -c "... $BASE ..."` — unexported vars silently vanish and the chain runs junk evals |

## 3. The winning config (cream cheese, task 1)

```
python -m eval.libero_eval --suite libero_object --task-ids 1 --n-trials 10 \
  --max-steps 600 --camera robot0_eye_in_hand_image --render-size 256 \
  --perception-period 2 --det-conf 0.02 --no-brake --role-disjoint-iou 0.1 \
  --source-max-area 0.12 \
  --checkpoint checkpoints/full_stageB_rec_fix.pt \
  --norm-stats data/libero_object_grid/norm_stats.json \
  --ibvs-phase --ibvs-gain 0.5 --ibvs-sign 1,-1,0 --ibvs-descend -0.4 \
  --ibvs-descend-hyst 0.50 --ibvs-target-uv 0.5,0.60 \
  --ibvs-grasp-offset 0.08,-0.05 --ibvs-close-z 0.01 --ibvs-press 0.2 \
  --ibvs-retry-rise 8 --ibvs-place-at=-0.005,0.257 --ibvs-drop-z 0.18 \
  --device cuda:0 --heads-device cpu --workers 1
```

### Calibrated constants (all offline, from logged runs / demo statistics)

| constant | value | source |
|---|---|---|
| grasp offset (lever arm) | (+0.080, −0.050) m | 231 at-gate episodes, band050 atlas |
| basket point (all 3 tasks) | (−0.005…−0.010, +0.255…0.260) | 50 demos/task end states |
| close_z: cream / soup / dressing | 0.010 / 0.045 / 0.114 | demo close heights (p10) |
| gate_z: cream / soup / dressing | 0.06 / 0.10 / 0.17 | object geometry |
| approach_z: cream / soup / dressing | 0 / 0.12 / 0.20 | fly-over height for standing objects |
| aim V: cream / soup / dressing | 0.60 / 0.60 / 0.24 | §5p |
| body mask | REVERTED (measured negative on cream) — finger-bind is a perception problem | §5t |

New `PhasedIBVS` params (all threaded through libero_eval + record_mp4):
`grasp_offset, close_z, press, retry_rise, yaw_probe/yaw_sign (built, unused
so far), place_at, drop_z, gate_z, approach_z`. Phases now:
`servo_src → align (probe k) → grasp → [rise→align]* → lift → transport →
release(lowered) → done`. Retries never re-consult vision (detector
unreliable at table height); first gate stores `_base_tgt`, probes shift it.

## 4. v4 failure taxonomy (what to fix for higher rate)

n=10: 2 success, 3 wrong-object binds at the visual gate (binding identity —
now THE dominant residual), 3 probe-exhausted (variance tail > ±6 cm), 2
timeouts while executing correctly (max-steps 400 → use 600). Grasp
competence 4/10. Ideas ranked: (a) clip-rerank / aspect gating at the gate
only (bind check before storing base_tgt); (b) widen/2-D probe; (c) verify
soup/dressing runs (in flight at write time: `logs/soup_v1.log`,
`logs/dressing_v1.log`, chain after 3 videos into
`eval_results/handeye_v5_vid`).

## 5. Defect 29 — check for siblings

Signed-mean of mirrored jaw joints ≡ 0. Fixed in both `_jaws()`. Grep for
other reducers over `proprio[7:9]` before writing new grasp logic. The test
mocks used same-sign jaw values and hid it — mocks now still same-sign;
consider a mirrored-jaw mock variant if touching this again.

## 6. Weight forensics (local, new)

`paper/weight_forensics.py` + `dynamic_forensics.py` + `render_ledger.py`
over the LOCAL checkpoint copies. 554-finding ledger + 24 figures. Highlights
that should drive next training run: planner's only live input is the
relational tokens (97% of ablation impact; fused/spatial/geometry/pred_box
are dead inputs); `drift.hrm.gain_head.weight` is ALL ZEROS (the learned
action-magnitude mechanism never trained — likely the §4p shrink root);
TRM dreaming is intrinsically stable (shared attractor, Jacobian σ₁=1.000);
tqsa text path is rank-28 (task-ID lookup). matplotlib was added to .venv
for this (documented deviation from the torch+numpy+pytest-only rule).

## 7. The distillation campaign (UNDERWAY — see UNAIDED_PLAN.md)

Four rounds run as of 2026-08-02 (recorder: `preprocess/teacher_rollouts.py`,
record + convert + DAgger mode; tests in `tests/test_teacher_rollouts.py`):

1. BC on 23 teacher successes → hover at 0.24 m, 0/3.
2. BC on 100 successes (`teacher_bc2`) → approach to 0.16 m then stall,
   0/10. Quantified: student |xy| 4–8× under teacher scale (mean-collapse
   under the teacher's hidden `_base_tgt`; `gain_head` still zero — its
   only gradient path is the actuation loss, which the proven recipe runs
   at weight 0).
3. DAgger (40 eps, student drives, β=0.3, teacher labels; NEW
   `--dagger-student-flags` in teacher_rollouts.py) + magnitude losses
   (`train/losses.py::pose_magnitude_loss`), but trained dagger-ONLY
   (`teacher_bc3`) → approach FIXED (0.061 m, stays at object) but
   grasping unlearned (labels ~all open), 0/10. Predicted in advance —
   see POD_COORDINATION.md.
4. `teacher_bc4` = aggregate (100 + 40) + magnitude losses
   (`scripts/round4_aggregate.sh`) → unaided_v4: IN FLIGHT at the last
   pod contact (launched 14:17 UTC; pod ssh then went down — chain is
   setsid-detached and survives blips; if the pod was RESTARTED, relaunch
   the script).

NOTE two Claude sessions share this pod — check POD_COORDINATION.md and
running processes before launching trainers/recorders.

## 8. Honesty rules (binding, unchanged)

Assisted ≠ unaided; constants are calibrated offline and disclosed; never
aggregate machine successes with policy numbers; `obj_pos` telemetry is
diagnosis-only, never a controller input.
