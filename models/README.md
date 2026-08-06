# Release manifest

Everything here is **measured, not transcribed**: digests were computed on the
files in this directory and, where noted, compared byte-for-byte against the
machine that produced the paper's cells. Anything that could not be verified is
marked absent rather than filled in.

## Checkpoint digests (SHA-256)

```
59a9fdee226efebf743de802f7cb896d05b438fb88ce5105564808f70c54c87b  full_stageB_rec_fix.pt
23139ff4f772b1eb6d2d11fc473f3245623c6719006b4864cfeaff3fc1367a84  gates_v1.pt
9ac9721134b781993e5c7eb3a2af31b290d5e01848366b30a97dd352e1c09680  goal_heads_v5.pt
e4e8925bd00d4cca300a71cec357b425f811ec0b18cfa2bbcdafd53e34a5f2e5  goal_heads_v7.pt
d155311a52975846e32d30bd3837c18e9e6799636d1e7f5e4ad9b7f7e59fe191  goal_heads_v8.pt
38e7611479b2d2a1cdaaa314def84620ce9612c0a03e6cb21b7303c875a85f19  role_bank.pt
522fba538f4f36ddf2ad48a27966a7d27a68e76b80b3ecf3df406bf6ed34326b  role_prototypes.pt
```

`goal_heads_v5.pt` and `full_stageB_rec_fix.pt` — the two that produce the
paper's headline cells — were verified **byte-identical** to the evaluation
machine's copies. A digest mismatch means you are not reproducing our numbers.

| file | what it is | params |
|---|---|---|
| `full_stageB_rec_fix.pt` | trunk (fusion + drift + planner) and world model | 7.0M + 9.97M |
| `goal_heads_v5.pt` | **flagship / released** goal heads (grasp + place) | 0.24M |
| `goal_heads_v7.pt` | multi-object head (addendum cells) | 0.24M |
| `goal_heads_v8.pt` | three-object head (adds a cream-cheese corpus) | 0.24M |
| `gates_v1.pt` | stage-1 learned gates (close trigger, hold check) | <3K |
| `role_bank.pt` | per-object crop-embedding bank, 1-NN role binding | 1432 vectors |
| `role_prototypes.pt` | mean class prototypes, the weaker binder | — |

`role_bank.pt` / `role_prototypes.pt` back binder cells that **did not**
improve the blocked object; see the App-D addendum, and note that their 0.902 /
0.613 figures are *offline* identity accuracy on corpus crops, which dissociates
from deployed success.

## Frozen detector

```
9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792  yolov8s-worldv2.pt
```

Not redistributed here (upstream Ultralytics weights, ~25.9 MB). This is the
only vision encoder *and* the only text encoder in the stack.

## Evaluated stack

Read off the evaluation machine, not from documentation:

| component | version |
|---|---|
| LIBERO | commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` (from source) |
| ultralytics | 8.4.115 |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| mujoco | 2.3.7 |
| robosuite | 1.4.1 |
| numpy | 2.2.6 |
| Python | 3.10 |

**The stack matters more than usual for this paper.** A rebuild of these
components shifted detector behaviour enough to invert which head scores
better (the audit-stack control: memorized 6/10 where the released head scores
0/10, the exact reverse of the deployment stack). Report these versions with
any reproduction attempt; a mismatch may be our own finding recurring rather
than a failed reproduction.

## Init-state hashes

Per-task SHA-256 digests of LIBERO-Object's shipped init-state arrays are in
`results/placement_pinning.json`, emitted by
`scripts/measure_placement_pinning.py` alongside the pinning measurement itself,
so the paper's opening claim is traceable to the exact init data it read.

## Reproducing

`paper/submission/REPRODUCE.md` — exact commands for both n=50 headline cells,
plus a `--mock-env` smoke test needing no LIBERO, no GPU, and no network.
Training commands are in the manuscript's §11 block. Runs made after commit
`faf832d` embed their own `argv` and git commit in `results.json`.
