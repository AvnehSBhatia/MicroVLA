# Release artifacts

What is in this repository, what each thing backs, and — where it matters — what
is *not* here and why. Written against the reviewer's release checklist (A1–A10)
so a reader can tell at a glance which claims are checkable and which rest on our
say-so.

## Instruments (run these; they need no checkpoints)

| file | what it measures | needs |
|---|---|---|
| `eval/probes.py` | both behavioural probes, with their inconclusive states | nothing |
| `scripts/measure_placement_pinning.py --mode direct` | what LIBERO-Object pins, per task, to the ULP | `libero` data files |
| `scripts/suite_forensics.py --pass columns` | placement entropy, all five suites | `libero` data files |
| `scripts/suite_forensics.py --pass joints` | per-object pinning via the compiled model | + mujoco |
| `scripts/attribution_profiles.py` | substitution attribution; the v2/v2.1 counterexample | corpus + heads |
| `scripts/probe_positive_control.py` | the identity-blindness probe's positive control | sim + detector |
| `scripts/position_vs_appearance.py` | the position-vs-appearance confound test | sim + detector |
| `eval/openvla_eval.py` | our probes against a public checkpoint | transformers + 15 GB ckpt |
| `scripts/analyze_pod_cells.py` | Wilson / exact McNemar / Newcombe over harvested trials | nothing |

## Trial-level results (A4)

`results/pod_cells.json` — every cell added in revision, with **per-trial**
outcomes keyed by trial index, its planned `n`, and a `complete` flag. The flag
is load-bearing: a partial read of a running batch is a *censored* sample, not a
small one, because successes end episodes early. `results/pod_run_ledger.txt`
gives per-job wall time and exit status for the same runs.

Older cells are cited by `run_id` in the manuscript's appendix and live on the
evaluation machine; those are **not** in this repository, which is a real gap
and is stated as one rather than papered over.

## Measurement artifacts

- `results/suite_forensics_{columns,joints}.json` — per-task placement stats and
  SHA-256 digests of every init-state array, all four suites
- `results/placement_pinning_direct.json` — the per-task pinning table
- `results/attribution_profiles.json` — the regenerated substitution profiles
- `results/probe_positive_control.json` — contrasts, `n_compared`, threshold sweep
- `results/pos_vs_appearance_{native,swap}.json` — the confound test, both arms

## Checkpoints and stack (A3)

`models/README.md` carries SHA-256 for every shipped checkpoint, the exact
evaluated stack **including `MUJOCO_GL`**, and the measurement showing why that
last one belongs there. `paper/submission/REPRODUCE.md` gives the two headline
cells as runnable commands.

## Figures

Every figure in the manuscript now has a generator under `paper/`, reading from
the JSON artifacts above. One historical exception is stated in its own caption:
`F2_attribution_profiles.png` predates its generator, and the numbers printed on
it are not reproduced by `scripts/attribution_profiles.py`. The qualitative
claims survive; the printed magnitudes should not be relied on.

## Not included, and why

- **The corpus** (`data/`, 540 episodes) — regenerable from the converters in
  `preprocess/`, and the repository is under a hard 10 GB budget.
- **The OpenVLA checkpoint** — 15 GB, upstream, fetched by name in
  `eval/openvla_eval.py`.
- **Pre-registration timestamps** — predictions are recorded in `paper/paper.md`
  in commit order, so `git log` dates them. That is weaker than a registry with
  external timestamps and we do not claim otherwise.
