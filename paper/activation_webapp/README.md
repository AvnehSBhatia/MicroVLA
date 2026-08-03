# MicroVLA Activation Observatory

## Model vs demo (local sim)

Side-by-side cream demo video + Three.js tabletop: orange arm integrates
`teacher_bc3` open-loop actions; blue ghost follows cream-cheese UV tracked
in the wrist pane (proxy — not logged proprio).

```bash
.venv/bin/python paper/activation_webapp/generate_demo_trace.py
.venv/bin/python - <<'PY'
# rebuild traj.json if needed — also written by tools below
PY
.venv/bin/python paper/activation_webapp/serve.py
# → http://127.0.0.1:8765/compare.html
```

## Demo cream mechinterp (open-loop)

Feed a dual-pane demo MP4 through the policy (wrist = right half), dump
activations + emitted actions, scrub next to the video. Primary default is
`teacher_bc3`; compare track is `rec_fix` on the same frames.

```bash
.venv/bin/python paper/activation_webapp/generate_demo_trace.py
.venv/bin/python paper/activation_webapp/serve.py
# → http://127.0.0.1:8765/demo.html
```

The MP4 has no logged actions — “choice difference” is primary vs compare
(and vs what the demo *looks* like).

## Soup success (preferred)

Four real LIBERO 1080p angles of alphabet-soup → basket, recorded on the box:

```bash
# on the pod
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa PYTHONPATH=/root/LIBERO \
  python -m eval.record_soup_angles \
    --checkpoint checkpoints/full_stageB_rec_fix.pt \
    --norm-stats data/libero_object_grid/norm_stats.json \
    --device cuda:0

# pull locally
scp -P 15266 -i ~/.ssh/id_ed25519 -r \
  root@213.173.107.139:/root/MicroVLA/paper/activation_webapp/data/soup_success \
  paper/activation_webapp/data/
```

Then open **http://127.0.0.1:8765/soup.html** (plain flowchart + angle switcher).

## Mock activation replay (legacy)

Interactive 30s replay of **every micro-module** under `full_stageB_rec_fix.pt`,
driven by a high-res LIBERO-like tabletop sim (Three.js). Mock perception only —
no LIBERO / cv2 / network.

## Generate the trace

```bash
.venv/bin/python paper/activation_webapp/generate_trace.py          # 900 ticks
.venv/bin/python paper/activation_webapp/generate_trace.py --ticks 30  # smoke
```

Writes `data/trace.json` (all leaf + composite module activations, plan/fused
heatmaps, planner input-withhold attribution, sim scene each tick). If the
checkpoint or norm stats are missing, the generator falls back to untrained
mock weights (`meta.weights = "mock"`) and `eval/identity_norm_stats.json`.

## Open the webapp

```bash
.venv/bin/python paper/activation_webapp/serve.py
# → http://127.0.0.1:8765/
```

Needs network once for the Three.js CDN import map. Trace JSON is local.

## What “contribution” means

Stacked bars are **planner input-withhold shares** (plan attribution on REAL
ticks), not LIBERO task success. Unaided closed-loop success can still be 0.
