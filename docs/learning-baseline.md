# First physical task baseline

The displayed `walk_imitation()` environment has no reference trajectory loaded.
Upstream `WalkImitation.get_reward_factors` returns `(1,)` in inference mode.
That reward cannot establish learning progress.

The separate `walk_on_ball()` environment rewards ball angular velocity near
`[0, -5, 0]`. Its body is tethered, so it isolates leg control from balance.
This is a candidate first motor experiment, not food-seeking or a language task.

Run `.venv/bin/python -m notflyguy.benchmark --steps 1000 --seeds 2`.
Each result covers 2 simulated seconds. Observed mean rewards:

| Policy | Seed | Mean reward |
| --- | --- | --- |
| Zero | 0 | 0.166546 |
| Random | 0 | 0.063236 |
| Zero | 1 | 0.166546 |
| Random | 1 | 0.061913 |

These are engineering baselines only, not trained performance. Random actuation
performs worse than doing nothing. A candidate must improve held-out reward and
move ball velocity toward the target; a nonzero reward alone is not evidence
of walking. Next: import a traceable connectome circuit, define modeled sensory
and motor projections explicitly, and train/evaluate against these controls.
