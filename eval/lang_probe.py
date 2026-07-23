"""Does the policy actually respond to the LANGUAGE instruction?

Splits "the model can't understand words" into two testable causes:

  (A) LANGUAGE DISCONNECTED (a bug) — the CLIP text-tower harvest returns
      near-constant embeddings, so every instruction yields the same text
      tokens and the model is literally blind to words.
  (B) LANGUAGE WIRED BUT WEAK (training) — text embeddings differ per
      instruction, but the BC-trained planner barely shifts its action.

Method: hold ONE frame fixed, vary only the instruction, and measure
  * pairwise cosine between the harvested (command) text embeddings, and
  * how much the emitted 7-DoF action changes across instructions.

If the TEXT embeddings are ~identical across very different instructions, the
harvest is broken (A). If they differ but the ACTIONS don't, language is wired
but the policy ignores it (B). Verdict is printed.

    PYTHONPATH=/root/LIBERO python -m eval.lang_probe \
        --checkpoint checkpoints/full_stageB.pt \
        --norm-stats data/libero/norm_stats.json --device cuda:0

No sim/mujoco needed — it builds the policy (real YOLO-World + CLIP text tower)
and runs it on a synthetic fixed frame, so it is fast. Optionally pass
``--frame path.png`` to probe on a real saved wrist frame instead.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch


# Deliberately diverse instructions: different objects, verbs, and
# destinations. If the action is ~identical across ALL of these on one frame,
# the policy is not using the words.
_INSTRUCTIONS = [
    "pick up the black bowl and place it on the plate",
    "pick up the red mug and place it in the basket",
    "push the wooden cabinet to the left",
    "pick up the milk carton and put it on the stove",
    "open the top drawer of the cabinet",
    "move the ketchup to the tray",
]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default="checkpoints/full_stageB.pt")
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frame", default=None, help="optional saved RGB frame (png/npy)")
    args = ap.parse_args(argv)

    from eval.policy import MicroVLAPolicy

    policy = MicroVLAPolicy(
        checkpoint=args.checkpoint, norm_stats=args.norm_stats, device=args.device
    )

    # One FIXED frame for every instruction. A synthetic frame has no
    # detectable objects (boxes fall back), which is fine: we are isolating the
    # TEXT pathway — same pixels every time, only the words change.
    if args.frame:
        if args.frame.endswith(".npy"):
            frame = np.load(args.frame)
        else:
            import cv2

            frame = cv2.cvtColor(cv2.imread(args.frame), cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame.astype(np.uint8))
    else:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    actions, text_embs = [], []
    for instr in _INSTRUCTIONS:
        policy.reset(instr)
        action = np.asarray(policy.act(frame), dtype=np.float32)
        actions.append(action)
        # The harvested (command) CLIP text embedding for this instruction.
        text_embs.append(policy.loop._task.command_emb.detach().clone())

    A = np.stack(actions)  # [N, 7]
    names = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "grip"]

    print(f"\nprobed {len(_INSTRUCTIONS)} instructions on one fixed frame "
          f"(checkpoint={policy.checkpoint_path}, stage_b={policy.is_stage_b})\n")

    print("emitted action per instruction (denormalized env units):")
    for instr, a in zip(_INSTRUCTIONS, A):
        print(f"  [{', '.join(f'{v:+.3f}' for v in a)}]  {instr}")

    # 1) Do the TEXT embeddings actually differ?
    text_cos = [
        _cos(text_embs[i], text_embs[j])
        for i, j in itertools.combinations(range(len(text_embs)), 2)
    ]
    mean_text_cos = float(np.mean(text_cos))
    max_text_cos = float(np.max(text_cos))

    # 2) Do the ACTIONS actually change across instructions?
    per_dim_std = A.std(axis=0)
    pose_spread = float(per_dim_std[:6].mean())      # mean std over the 6 pose dims
    grip_vals = set(np.sign(A[:, 6]).tolist())        # {-1}, {+1}, or both

    print("\n--- text pathway ---")
    print(f"  mean pairwise cosine of (command) text embeddings: {mean_text_cos:.3f} "
          f"(max {max_text_cos:.3f})")
    print("  (near 1.0 => the CLIP harvest returns ~constant text => language is "
          "DISCONNECTED / bugged)")

    print("\n--- action response ---")
    print("  per-dim action std across instructions:")
    for nm, s in zip(names, per_dim_std):
        print(f"    {nm:8s} {s:.4f}")
    print(f"  mean pose-dim spread: {pose_spread:.4f}")
    print(f"  gripper signs seen across instructions: {sorted(grip_vals)}")

    print("\n=== verdict ===")
    if mean_text_cos > 0.98:
        print("  (A) LANGUAGE DISCONNECTED: text embeddings are ~identical across "
              "very different instructions. The CLIP text-tower harvest is broken "
              "(eval/policy.py -> ClipTaskEncoder). Fix the harvest FIRST; no "
              "amount of grounding/planner work matters until words produce "
              "distinct embeddings.")
    elif pose_spread < 1e-3:
        print("  (B) LANGUAGE WIRED BUT IGNORED: text embeddings DO differ, but the "
              "action barely moves across instructions. The planner learned to lean "
              "on non-language cues (classic MSE-BC collapse). Levers: role-ordered "
              "grounding (Feature 1) so the clause moves the grounded box, stronger "
              "text conditioning, more/balanced data.")
    else:
        print("  LANGUAGE IS RESPONSIVE: both text embeddings AND actions vary with "
              "the instruction. 'Can't understand words' is more likely precision / "
              "grounding of the RIGHT object than a language disconnect — compare "
              "libero_spatial t0/t8 after the grounding fix.")


if __name__ == "__main__":
    main()
