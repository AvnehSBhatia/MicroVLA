"""Run our probes against a public VLA we did not build (referee E2 / M2).

Every instrument in this paper has been demonstrated on exactly one stack --- our
own --- which makes each claim of portability an aspiration. This runs the two
behavioural instruments against ``openvla/openvla-7b-finetuned-libero-object``,
a checkpoint from a different lab, a different architecture (7B autoregressive
VLM, no detector, no engineered shell) and a different action convention:

  ``--protocol shipped``   the suite as published: shipped init states.
  ``--protocol randomized`` our +-R cm source teleport, same draws as our own
                            randomized cells.
  ``--protocol swap``       our instruction-swap probe: tell the policy a
                            different task's instruction, score success on the
                            REAL task. Identical decision rule to
                            ``eval/probes.py::instruction_swap``.

The identity-blindness probe is NOT run here and cannot be: it compares which
*detection box* two prompt chains select, and OpenVLA has no detector and no box
to compare. That is a scope limit of the instrument, not a result, and the paper
says so rather than substituting something else and calling it the same probe.

Deliberately kept out of ``eval/policy.py`` and off the deployment stack: this
needs ``transformers``, a 7B checkpoint and its own pinned environment, none of
which may perturb the versions the paper's own cells depend on. Run it with the
separate interpreter that has those deps.

Usage:
    python eval/openvla_eval.py --protocol shipped --n-trials 10 --task-ids 0 \
        --checkpoint /workspace/openvla-libero-object --out results/ovla_shipped.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: OpenVLA's LIBERO fine-tunes were trained on 256x256 agentview frames. This is
#: their protocol, not ours; running them on our wrist camera would measure a
#: distribution shift we invented rather than the policy.
CAMERA = "agentview_image"
RENDER = 256


def build_policy(ckpt: str, device: str):
    """The published inference path: AutoModelForVision2Seq + its own processor."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def predict(processor, model, frame: np.ndarray, instruction: str, device: str,
            unnorm_key: str) -> np.ndarray:
    """One 7-DoF action. Prompt format is OpenVLA's own, not ours."""
    import torch
    from PIL import Image

    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(frame).convert("RGB")).to(
        device, dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    a = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    # OpenVLA's own LIBERO post-processing, and it is load-bearing: the model
    # emits the gripper in [0, 1] with 1 = CLOSE, while robosuite wants
    # [-1, +1] with -1 = close. Skipping it leaves the jaw open for the whole
    # episode -- the policy never grasps and scores 0, which we measured before
    # adding this and would have published as OpenVLA's baseline.
    a[-1] = np.sign(2.0 * a[-1] - 1.0)      # normalize_gripper_action(binarize)
    a[-1] = -a[-1]                           # invert_gripper_action
    return a


def run(args) -> dict:
    from eval._libero_compat import prepare_libero

    prepare_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    from eval.libero_eval import randomize_source_xy

    bench = benchmark.get_benchmark_dict()[args.suite]()
    processor, model = build_policy(args.checkpoint, args.device)

    task_ids = [int(t) for t in args.task_ids.split(",")] if args.task_ids else list(range(10))
    out: dict = {"suite": args.suite, "protocol": args.protocol,
                 "checkpoint": args.checkpoint, "camera": CAMERA,
                 "max_steps": args.max_steps, "seed": args.seed,
                 "randomize_source_xy": args.randomize_source_xy,
                 "override_instruction": args.override_instruction,
                 "argv": sys.argv, "per_task": {}, "trials": []}

    for ti in task_ids:
        task = bench.get_task(ti)
        bddl = str(Path(bench.get_task_bddl_file_path(ti)))
        init_states = np.asarray(bench.get_task_init_states(ti))
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=RENDER, camera_widths=RENDER)
        succ = 0
        try:
            for trial in range(args.trial_offset, args.trial_offset + args.n_trials):
                # SAME trial->state map as eval/libero_eval.py, so a cell here is
                # comparable to the same-named cell there.
                trial_seed = args.seed * 1_000_003 + trial
                env.seed(trial_seed)
                env.reset()
                obs = env.set_init_state(init_states[trial_seed % len(init_states)])
                if args.randomize_source_xy > 0:
                    randomize_source_xy(env, np.random.default_rng(777_000 + trial_seed),
                                        args.randomize_source_xy)
                    inner = getattr(env, "env", env)
                    obs = inner._get_observations(force_update=True)
                # The swap: the policy is TOLD a different task; the env, its
                # scene and its success predicate are untouched, so success is
                # still scored on the real task.
                instr = args.override_instruction or task.language
                t0, done = time.time(), False
                # Their eval lets the scene settle before the policy acts; acting
                # into a still-falling scene is not the protocol the checkpoint
                # was evaluated under.
                for _ in range(args.settle_steps):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                for step in range(args.max_steps):
                    # 180-degree rotation, NOT a vertical flip. This is
                    # OpenVLA's own LIBERO preprocessing
                    # (experiments/robot/libero/libero_utils.py::get_libero_image);
                    # feeding their policy our convention would hand it images
                    # its training never saw and manufacture a weak baseline,
                    # which would flatter our probe rather than test it.
                    frame = np.ascontiguousarray(obs[CAMERA][::-1, ::-1])
                    a = predict(processor, model, frame, instr, args.device, args.unnorm_key)
                    obs, _, done, _ = env.step(a.tolist())
                    if done:
                        break
                succ += int(bool(done))
                out["trials"].append({"task": task.name, "trial": trial,
                                      "success": bool(done), "steps": step + 1,
                                      "instruction_given": instr,
                                      "seconds": round(time.time() - t0, 1)})
                print(f"[ovla] {task.name[:44]:<46} trial {trial}: success={bool(done)} "
                      f"steps={step+1} ({time.time()-t0:.0f}s)", flush=True)
        finally:
            env.close()
        out["per_task"][task.name] = succ / args.n_trials
        print(f"[ovla] {task.name[:44]:<46} {succ}/{args.n_trials}", flush=True)

    k = sum(t["success"] for t in out["trials"])
    n = len(out["trials"])
    out["k"], out["n"] = k, n
    out["mean_success"] = k / n if n else 0.0
    print(f"\n[ovla] {args.protocol}: {k}/{n} = {out['mean_success']:.3f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/workspace/openvla-libero-object")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-ids", default="0")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--settle-steps", type=int, default=10,
                   help="dummy steps before the policy acts, matching OpenVLA's "
                        "own LIBERO eval (num_steps_wait).")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--unnorm-key", default="libero_object")
    p.add_argument("--protocol", choices=["shipped", "randomized", "swap"],
                   default="shipped")
    p.add_argument("--randomize-source-xy", type=float, default=0.0)
    p.add_argument("--override-instruction", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.protocol == "randomized" and a.randomize_source_xy <= 0:
        a.randomize_source_xy = 0.04
    if a.protocol == "swap" and not a.override_instruction:
        a.override_instruction = "pick up the butter and place it in the basket"
    res = run(a)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
