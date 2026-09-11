# First physical task baseline

## The task

The displayed `walk_imitation()` environment has no reference trajectory loaded.
Upstream `WalkImitation.get_reward_factors` returns `(1,)` in inference mode.
That reward cannot establish learning progress.

The separate `walk_on_ball()` environment rewards ball angular velocity near
`[0, -5, 0]`. Its body is tethered, so it isolates leg control from balance.
This is a candidate first motor experiment, not food-seeking or a language task.

Reward is the product of three linear tolerance factors, so lateral ball
velocity penalises the score multiplicatively. Approaching `1.0` needs a
coordinated gait, not a held posture.

## Corrections to earlier assumptions

Two facts about this environment were wrong in the previous revision of this
document and are recorded here because they change what training can do.

**The action spec is not `[0, 1]`.** Only the head and abdomen actuators span
`[0, 1]`; leg actuators span their own ranges — `coxa_T1_left` is `[-3, 3]`.
Measured unique bounds across the 59 actuators:

| | values |
| --- | --- |
| `action_spec().minimum` | −3, −1.35, −1.05, −1, −0.9, −0.8, −0.75, −0.7, −0.5, −0.3, −0.2, −0.15, 0 |
| `action_spec().maximum` | 0.2, 0.25, 0.3, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.5, 1.7, 1.8, 2, 3 |

A readout with a fixed `(0, 1)` output map therefore cannot reach most of the
posture space: all-ones scores `0.000`, while the spec-maximum posture scores
`0.204`. The readout now maps affinely onto each actuator's declared range
(`motor_projection.limits_from_spec`), so a zero drive lands on that actuator's
own midpoint. This single correction raised the **untrained** controller's
reward from `0.000` to `0.1465`, against a zero-action baseline of `0.1661`.

**Seeds are degenerate.** `walk_on_ball()` performs no episode randomization in
this build: the zero-policy reward was bit-identical at `0.1660720560` for
seeds 0, 1, 2, 10, 11, 13 and 42, with identical initial `qpos`. "Held-out
seeds" are nominal only. The genuine held-out axis available here is episode
length (200 training steps versus the full 1000-step episode), and real
generalization would require domain randomization or perturbed initial states.

## Measured baselines

Command: `.venv/bin/python -m notflyguy.benchmark --steps 200 --seeds 2`

Current results (200 steps = 0.4 simulated seconds, seed 0):

| Policy | Action | Mean reward |
| --- | --- | --- |
| `zero` | every actuator at its minimum (0.0) | **0.16607** |
| `neutral` | every actuator at 0.5 | 0.00000 |
| `max` | every actuator at its spec maximum | 0.20410 |
| `random` | uniform over the spec | 0.047–0.059 |
| untrained readout | per-actuator midpoint | 0.14652 |

The all-zeros command is a strong local optimum and remains the number to beat.
Note that `neutral` (0.5) is *worse* than `zero` because 0.5 is not mid-range
for leg actuators whose spec straddles zero.

## Where training stands

`notflyguy/train.py` runs a cross-entropy method (CEM) over the readout. The
best completed evidence is a **2-iteration smoke run**, evaluated on a nominal
held-out seed:

| | reward |
| --- | --- |
| zero baseline | 0.1660720560 |
| candidate | 0.1661388199 |
| delta | **+0.0000668** |

That is not a real margin, and the report marks it as such
(`beats_zero_by_margin_0.01: false`). A 30-iteration CEM run using the `bias`
parameterization from a `neutral` initialization reached a **training-horizon
elite mean of 0.5579** (best sample 0.5975) at iteration 16 against a zero
baseline of 0.16607, but it was stopped before any evaluation was written. Those
are training scores on training seeds over 20% of an episode — **not** a result.

Independent evidence that a real win is reachable: 80 uniformly random postures
drawn inside the readout's reachable set produced a best score of `0.3785`
(mean ball velocity `[0.23, -2.96, -2.21]`). Good postures exist in the space;
no run has yet demonstrated held-out improvement.

An important failure mode, recorded so it is not repeated: initializing CEM at
the `zero` policy (bias ≈ −4) freezes the search. `tanh` is saturated there, the
readout has almost no control authority, and the optimizer is blind. Initialize
at `neutral` with a nonzero bias standard deviation.

Next: import a traceable connectome circuit (done), define modeled sensory and
motor projections explicitly (done), and complete a held-out evaluation of a
trained candidate against the controls above (outstanding).
