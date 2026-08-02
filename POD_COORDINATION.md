# Pod coordination note (2026-08-02, session B)

Two Claude sessions are working this pod. Current division:

- `scripts/unaided_dagger_chain.sh` (session A) is RUNNING: DAgger record
  40 eps -> convert -> train teacher_bc3 (magnitude losses) -> unaided_v3.
- Session B killed its duplicate round3 chain and the v2r/diag evals
  (CPU contention, 6x slowdown); evals rerun during the train window via
  `scripts/evals_v2r_diag.sh`.

INTERVENTION PLANNED by session B at the train stage: the chain trains on
`data/teacher_dagger_soup_grid` ALONE, but DAgger labels there are ~always
grip-open (teacher never reaches grasp under beta=0.3 student driving:
12/12 failures, label grip mean -0.976). Dagger-only training would
unlearn grasp/place. Fix: train on the AGGREGATE
`--data-dir data/teacher_grid2 data/teacher_dagger_soup_grid`, same
magnitude-loss flags. If you (session A) get there first, do the same and
note it here. Do not run two trainers at once.

Label sanity (first 6 dagger eps, 3600 ticks): teacher-label mean|xyz|
0.103/0.193/0.268 raw units == round-2 teacher stats; executed mix 0.294
teacher fraction == beta. Data is GOOD, coverage is approach-corridor.
