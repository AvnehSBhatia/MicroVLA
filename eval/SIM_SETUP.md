# LIBERO / robosuite on macOS ARM — installability findings (E3 risk retirement)

Machine: Apple Silicon (Darwin 25.6, arm64), Python 3.13.2, venv at `.venv`.
Pre-existing venv: `torch==2.13.0`, `tensorflow==2.21.0`, `numpy==2.5.1`,
`opencv-python==5.0.0.93` (do not reinstall/upgrade torch or tensorflow).

## TL;DR

- **mujoco + robosuite install cleanly on macOS ARM and render headlessly.**
  This retires the historical "no EGL on macOS" worry: `MUJOCO_GL=cgl` (and
  `MUJOCO_GL=glfw`, windowless) both work for offscreen rendering. Verified
  end-to-end with a real robosuite `Lift` env: `reset()`, 5 random actions,
  and `robot0_eye_in_hand_image` grabbed successfully (`(128, 128, 3) uint8`).
- **LIBERO itself is only partially viable in this session**, blocked by two
  independent, well-understood issues — not macOS-ARM incompatibility:
  1. The PyPI `libero` package (`libero==0.1.1`, HF's mirror of
     `Lifelong-Robot-Learning/LIBERO`) pins `robosuite==1.4.0`. Its env code
     imports `robosuite.environments.manipulation.single_arm_env`, a module
     that no longer exists in robosuite 1.5.2 (moved during a refactor).
     Actually instantiating a LIBERO env therefore needs robosuite downgraded
     to 1.4.0, which was **not attempted** — see "Not attempted" below.
  2. Neither the PyPI wheel nor a plain `pip install` of anything ships the
     LIBERO 3D assets (meshes/textures). They are fetched on first use via
     `huggingface_hub.snapshot_download` (a genuine network dependency) or
     are present in-repo only if you `git clone` the full
     `Lifelong-Robot-Learning/LIBERO` repo, whose `libero/libero/assets/`
     directory alone is **404 MB** (measured via a real sparse checkout).
     Both routes are excluded by this task's constraints (no large
     downloads; keep total added site-packages under ~1 GB — mujoco +
     robosuite + their transitive deps already consume ~900 MB of that
     budget on their own).
- The **benchmark registry** (task suites, BDDL files, language
  instructions — no physics, no assets needed) works fully: all 6 suites
  (`libero_spatial`, `libero_object`, `libero_goal`, `libero_90`,
  `libero_10`, `libero_100`) enumerate correctly and BDDL files resolve to
  real paths on disk.

Net effect for E3: use `eval/libero_eval.py`'s `mock_env=True` path (already
in the shared contract) for the CPU-only, no-network, no-heavy-deps default
harness everywhere. On a machine with more disk/network budget (or a Linux
CI box, see below), swap in the real env using the exact recipe below —
the harness code doesn't need to change, only how you construct the env
object passed to `policy.act()`.

## Exact working install commands (macOS ARM, this venv)

```bash
# 1. mujoco — installs clean, no conflicts
.venv/bin/pip install mujoco==3.3.0
#    NOTE: do NOT take the latest mujoco (3.10.0 as of this writing). Its
#    Python binding changed mj_fullM()'s signature from
#    (model, dst, sparse_M) to (model, data, dst); robosuite 1.5.2's
#    controller code still calls the old 3-positional-array form and raises
#    `TypeError: mj_fullM(): incompatible function arguments` at env reset.
#    mujoco==3.3.0 (robosuite's declared floor, mujoco>=3.3.0) has the old
#    signature and works.

# 2. robosuite — pulls numba/scipy/mink/pyobjc etc; downgrades numpy
.venv/bin/pip install robosuite==1.5.2
# 3. Re-pin numpy — robosuite's resolver drags numpy down to 1.26.4 via
#    numba's `numpy<2.5` ceiling, but tensorflow's ml-dtypes/dm-tree need
#    numpy>=2.1.0. 2.4.x is the only band that satisfies both.
.venv/bin/pip install "numpy==2.4.6" --no-deps
# then re-pin mujoco (installing robosuite pulls its own mujoco floor build,
# repeat step 1 if pip resolved a newer mujoco during step 2 — check with
# `pip show mujoco`).

# 4. LIBERO — PyPI package only (see caveats above). --no-deps to avoid
#    pulling training-only deps (transformers, wandb, tensorboard,
#    robomimic, hydra-core, thop) that libero.libero (env/benchmark code)
#    never imports — those are only used by the libero.lifelong subpackage.
.venv/bin/pip install --no-deps libero==0.1.1 bddl==1.0.1
.venv/bin/pip install pyyaml cloudpickle gymnasium jupytext future
#    (jupytext is bddl's own dependency, used for its .bddl file tooling;
#    future is needed by bddl/backend_abc.py: `from future.utils import
#    with_metaclass`.)
```

Total added to `.venv/lib/python3.13/site-packages`: **~900 MB**
(`du -sh .venv` went 2.5G -> 3.4G). Biggest single contributor is
robosuite itself (528 MB — it bundles its own MJCF robot/gripper/arena
assets), then numba+llvmlite (154 MB), scipy (97 MB), mujoco (53 MB).

### Env vars

```bash
export MUJOCO_GL=cgl      # or: glfw   (both verified; osmesa/egl are NOT
                           # available on macOS — they raise
                           # "invalid value for environment variable
                           # MUJOCO_GL" at import time)
export LIBERO_CONFIG_PATH=/some/writable/dir   # see pitfall below
```

## Smoke test — what actually runs today (robosuite only, CPU-only)

```python
import numpy as np, robosuite as suite
env = suite.make(
    env_name="Lift", robots="Panda",
    has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
    camera_names=["robot0_eye_in_hand"], camera_heights=128, camera_widths=128,
    control_freq=20,
)
obs = env.reset()
img = obs["robot0_eye_in_hand_image"]          # (128, 128, 3) uint8
low, high = env.action_spec                    # both shape (7,), range [-1, 1]
rng = np.random.default_rng(0)
for _ in range(5):
    obs, reward, done, info = env.step(rng.uniform(low, high))
env.close()
```
Result (both `MUJOCO_GL=cgl` and `MUJOCO_GL=glfw`): runs to completion,
`img.shape == (128, 128, 3)`, `img.dtype == uint8`, pixel range `[9, 254]`
(non-degenerate — camera is actually seeing the scene), `action_spec` is
`(low=[-1]*7, high=[1]*7)` (7-DoF: 6 delta-pose + 1 gripper, matches
`cfg.num_servos` incidentally though semantics differ — LIBERO/robosuite's
7 dims are OSC delta-pose + gripper, not raw per-servo PWM).

`obs.keys()` for `Lift`/Panda: `robot0_eye_in_hand_image`, plus proprio
(`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, joint pos/vel/
acc, sin/cos encodings) and task state (`cube_pos`, `cube_quat`,
`object-state`). LIBERO envs (`libero.libero.envs.OffScreenRenderEnv`) wrap
the same robosuite observation dict — same `robot0_eye_in_hand_image` key —
plus BDDL-defined object/region states, once the version conflict below is
resolved.

## LIBERO API entry points (registry — verified working, no assets needed)

```python
import os
os.environ["LIBERO_CONFIG_PATH"] = "/tmp/libero_home"   # see pitfall below
from libero.libero import benchmark
bm_dict = benchmark.get_benchmark_dict()
# {'libero_spatial': ..., 'libero_object': ..., 'libero_goal': ...,
#  'libero_90': ..., 'libero_10': ..., 'libero_100': ...}
suite_obj = bm_dict["libero_spatial"]()   # class name: LIBERO_SPATIAL
suite_obj.n_tasks                          # 10
task = suite_obj.get_task(0)               # task.name, task.language (instruction str)
bddl_path = suite_obj.get_task_bddl_file_path(0)   # real file, exists on disk

# Env construction (once robosuite==1.4.0 + assets are available):
from libero.libero.envs import OffScreenRenderEnv
env = OffScreenRenderEnv(bddl_file_name=bddl_path,
                          camera_heights=128, camera_widths=128)
obs = env.reset()   # same robot0_eye_in_hand_image key as plain robosuite
```

paper.md's three E3 suites (`libero_spatial`/`libero_object`/`libero_goal`)
all exist in the registry today and enumerate 10 tasks each with real BDDL
files and human-language instructions — task/instruction pairing needs no
further scaffolding once an env can be instantiated.

## Pitfalls hit (in the order encountered)

1. **`pip install robosuite` silently downgrades `numpy` to 1.26.4**, which
   then breaks `tensorflow`'s `ml-dtypes`/`dm-tree` version floor
   (`numpy>=2.1.0`) — pip's resolver doesn't cross-check already-installed
   packages it isn't touching. Verified this doesn't actually crash imports
   at 1.26.4 (tf/torch/cv2/microvla all still imported fine, full
   `pytest tests` suite — 115/115 at that point — still passed), but it's a
   latent conflict, not a real fix. Re-pinned to `numpy==2.4.6`, the one
   band that satisfies numba's `<2.5` ceiling and tensorflow's `>=2.1.0`
   floor simultaneously. Re-verify after any resolver-driven install: `pip
   install robosuite` / `pip install libero` will each try to re-drag numpy
   around and print (harmless-looking, but check them) "incompatible"
   warnings.
2. **`mujoco` latest (3.10.0) breaks robosuite 1.5.2's mass-matrix
   controller call** (`mj_fullM()` signature change — see above). Only
   surfaces at `env.reset()`, not at import time, and the traceback prints
   a huge embedded array repr that looks like a crash dump — it is not; the
   real error is `TypeError: mj_fullM(): incompatible function arguments`
   near the bottom. Fix: pin `mujoco==3.3.0`.
3. **`osmesa` / `egl` are not valid `MUJOCO_GL` values on macOS** — mujoco's
   Python package raises `RuntimeError: invalid value for environment
   variable MUJOCO_GL` at import time for either. Only `glfw` (windowless
   GLFW context) and `cgl` (Apple's CGL, no window server needed) work.
   Both were verified to produce real, non-blank offscreen renders.
4. **`libero.libero/__init__.py` blocks on `input()`** the first time it's
   imported with no config file present, asking interactively whether you
   want a custom dataset path — this hangs/`EOFError`s under any
   non-interactive driver (pytest, CI, subprocess). Fix: set
   `LIBERO_CONFIG_PATH` to a writable directory and pre-write
   `config.yaml` yourself before the first import:
   ```yaml
   benchmark_root: <path-to-site-packages>/libero/libero
   bddl_files: <...>/libero/libero/bddl_files
   init_states: <...>/libero/libero/init_files
   datasets: <...>/libero/libero/../datasets
   assets: <...>/libero/libero/assets
   ```
5. **`bddl==1.0.1`'s `install_requires` includes `jupytext`** (for its BDDL
   notebook tooling, unrelated to parsing at runtime) which drags in
   `nbformat`/`jsonschema`/`rpds-py`/`traitlets`/`jupyter-core` — all small,
   harmless, but unexpected for a "just parse this DSL" package.
6. **`bddl/backend_abc.py` imports `future.utils`** (`from future.utils
   import with_metaclass`) — `future` isn't in `bddl`'s own
   `install_requires` even though the import is unconditional at package
   load. Needed as an explicit extra install.
7. **The PyPI `libero` wheel ships zero mesh/texture assets.**
   `libero/libero/assets/` does not exist after `pip install`; only code +
   BDDL task files + init-state files are packaged. `get_assets_path()`
   falls back to `huggingface_hub.snapshot_download` at first env-creation
   time if the local folder is missing — a real, uncapped network fetch,
   which this task's constraints forbid attempting.
8. **`libero==0.1.1` hard-pins `robosuite==1.4.0`**, and it's not just a
   version-floor formality: `libero/libero/envs/bddl_base_domain.py`
   imports `robosuite.environments.manipulation.single_arm_env`, a module
   robosuite 1.5.2 removed/relocated during a refactor
   (`ModuleNotFoundError`). Confirmed this only manifests when actually
   importing `libero.libero.envs` (env/task classes) — the benchmark
   registry (`libero.libero.benchmark`) has no robosuite import chain at
   all and works regardless of robosuite version.
9. `robosuite` also declares `mink==0.0.5` (whole-body IK for humanoid
   robots), which itself requires `numpy<2.0.0` — directly incompatible
   with the `numpy==2.4.6` pin above. robosuite handles this gracefully at
   import (`[robosuite WARNING] Could not load the mink-based whole-body IK
   ...`) and it never surfaces for arm-only robots (Panda et al. — the only
   robots relevant here). Left as a known, inert conflict; not worth
   chasing since it never touches our code path.

## Not attempted (would need more disk/network budget than this task allows)

- **Downgrading `robosuite` to `1.4.0`** to satisfy LIBERO's pin. This
  wasn't tried because (a) it would very likely reopen the exact same
  `mj_fullM`-style mujoco-binding version dance from scratch (robosuite
  1.4.0 predates the mujoco>=3.3.0 API robosuite 1.5.2 uses, so the working
  `mujoco==3.3.0` pin above is not guaranteed to still be compatible), and
  (b) it wouldn't unblock the asset problem anyway (item 7), so the payoff
  didn't justify the added risk to an otherwise-working, verified
  robosuite 1.5.2 install. If a later session needs full LIBERO physics
  specifically (not just plain robosuite), do this in a **separate venv**
  to avoid destabilizing the config validated here.
- **Downloading the real asset tree.** Confirmed via a `git clone
  --filter=blob:none --sparse` of `Lifelong-Robot-Learning/LIBERO` limited
  to `libero/libero` that assets alone are 404 MB
  (`scenes` 105M, `stable_hope_objects` 100M, `stable_scanned_objects` 85M,
  `turbosquid_objects` 69M, `textures` 35M, `articulated_objects` 10M).
  That clone was done in `/tmp` purely to measure size and was deleted
  immediately after — nothing from it was installed or copied into the
  venv or repo.

## Is it viable on macOS ARM? Yes for the physics/rendering stack; not (in this session) for LIBERO's full asset tree

This is **not** an ARM compatibility problem — mujoco + robosuite install
and render headlessly on Apple Silicon with zero issues once pinned
correctly (`mujoco==3.3.0`, `MUJOCO_GL=cgl`). The blockers are (a) a
robosuite API-version pin mismatch specific to `libero==0.1.1`'s current
release, and (b) a 404 MB asset payload that has to come from somewhere
(network or a big git clone) — both would apply equally on Linux. If E3
needs real LIBERO physics rather than `mock_env=True`, the cheapest path is
a Linux box (or this same Mac with a relaxed disk/network budget) running:

```bash
python -m venv .venv-libero   # SEPARATE venv, don't reuse this one
.venv-libero/bin/pip install "robosuite==1.4.0"
.venv-libero/bin/pip install --no-deps libero==0.1.1 bddl==1.0.1
.venv-libero/bin/pip install pyyaml cloudpickle gymnasium jupytext future \
    hydra-core easydict thop hf-egl-probe
# mujoco version: start from whatever `pip install robosuite==1.4.0` pulls
# in and only override it if env.reset() throws a mj_fullM-style
# TypeError (see pitfall 2) — verify empirically, don't assume 3.3.0
# still applies to the 1.4.0 controller code path.
export LIBERO_CONFIG_PATH=~/.libero   # or anywhere writable
export MUJOCO_GL=cgl   # macOS; on Linux use egl or osmesa instead
python -c "from libero.libero import get_libero_path"  # triggers first-run
# asset download via huggingface_hub.snapshot_download (~400+ MB, network)
```
