# Reproducing the headline cells

Every number in the paper is cited by a `run_id` that names a JSON artifact.
This file gives the exact command for the two cells the abstract leads with, the
checkpoint hashes they were produced with, and what to expect when you run them.

## What you need

LIBERO with a working MuJoCo offscreen renderer, plus this repo. The evaluation
does not need training data, a network connection, or the training stack — only
the two checkpoints below, both tracked in this repository.

```
md5  01ff8728aa5ee582819372639a5ec695  models/goal_heads_v5.pt          (0.24 M, the released "flagship" head)
md5  a8ea1cda8a6c50bf9db7f8585a2994c9  models/full_stageB_rec_fix.pt    (trunk + world model)
```

These hashes are the weights that produced the numbers below, verified identical
between this repository and the machine that ran them. If your copies hash
differently, you are not reproducing the paper's cells.

## Smoke test first (no LIBERO, no GPU, no network)

```bash
.venv/bin/python -m eval.libero_eval --mock-env --n-trials 2 --task-ids 0 \
  --out-dir /tmp/microvla_smoke
```

This exercises the whole harness against a mock environment. It proves your
install and the results-writing path work before you spend GPU-hours. It does
**not** produce a paper number — the mock backend is not LIBERO.

## The two headline cells

Both use one task (`libero_object` task 0, alphabet soup), the wrist camera, and
600 steps. The only difference is the protocol: held-out draws seed 20;
randomized draws seed 0 and teleports the source object ±4 cm.

```bash
export PYTHONPATH=/path/to/LIBERO:$PWD
export MUJOCO_GL=osmesa

FLAGS="--checkpoint models/full_stageB_rec_fix.pt \
  --norm-stats eval/identity_norm_stats.json \
  --camera robot0_eye_in_hand_image --render-size 256 \
  --perception-period 2 --det-conf 0.02 --no-brake \
  --role-disjoint-iou 0.1 --source-max-area 0.12 \
  --device cuda:0 --heads-device cpu --workers 1 \
  --goal-ckpt models/goal_heads_v5.pt"

# Held-out, n=50  -> expect mean_success 0.700  (35/50, Wilson [0.56, 0.81])
python -m eval.libero_eval --suite libero_object --task-ids 0 \
  --n-trials 50 --max-steps 600 --seed 20 $FLAGS --out-dir eval_results/repro_heldout

# Randomized +-4 cm, n=50 -> expect mean_success 0.520 (26/50, Wilson [0.39, 0.65])
python -m eval.libero_eval --suite libero_object --task-ids 0 \
  --n-trials 50 --max-steps 600 --seed 0 $FLAGS --randomize-source-xy 0.04 \
  --out-dir eval_results/repro_rand
```

Reference artifacts: `libero_object_real_1785899388619` (held-out, `mean_success`
0.7, `n_trials` 50) and `libero_object_real_1785904148049` (randomized, 0.520).

## What "reproduce" honestly means here

**Expect the interval, not the point.** These are Bernoulli cells at n=50. Even a
bit-identical rerun of a *different* seed stream should land inside the Wilson
interval, not on 0.700 exactly. Treat a result inside [0.56, 0.81] as a
reproduction and a result outside it as a discrepancy worth reporting.

**Physics is not bit-reproducible across stacks.** The paper's central control
finding is that a MuJoCo/robosuite rebuild can shift detector behaviour enough to
invert which head looks better (§ the audit-stack control: the memorized head
scores 6/10 where the released head scores 0/10, the exact reverse of the
deployment stack). So a mismatch on a *different* software stack is a
scientifically interesting result, not necessarily a failed reproduction — it is
the paper's own finding recurring. Report the stack versions with any mismatch.

**One task, one object.** Every claimed cell is a single task and object at
n=10, with these two confirmed at n=50. Nothing here is comparable to
community-protocol LIBERO scores, which use different step budgets, cameras, and
training corpora.

## Provenance going forward

`results.json` now records `provenance` — the full `argv`, the git commit, and
whether the tree was dirty. The two headline artifacts above **predate** this
field, which is why their commands had to be recovered from the shell scripts
that launched them; that recovery is the reason the field exists. Runs made
after commit `faf832d` are self-describing.
