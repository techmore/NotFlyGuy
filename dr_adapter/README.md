# DragonRealms adapter

This is a removable, local, advisory-only adapter. It does not import NotFlyGuy internals or modify DragonRealms. The default transport is a mock; no keyboard automation or client writes exist.

Run from the Fly project:

```bash
python -m dr_adapter.dashboard
```

To display the saved training ledger, add
`--training-state data/dr-runs/training-state.json`. The page at
http://127.0.0.1:8787/ shows episode totals, reward components, freshness,
observer telemetry, and expiring proposals. Missing data stays unavailable.
Observer controls apply only to the dashboard session, not a separately
running `dr_adapter.pilot` process. Resume clears the old proposal and waits
for a fresh observation. Emergency stop requires a process restart to re-arm.
Training update counts are calls to the learning rule, not a measure of
improved performance; the recurrent connectome remains fixed.

Reward version `balanced-requirements/2` scores reductions in the mean squared
normalized requirement deficit, plus a bonus for closing a requirement row.
This favors lagging requirements and stops rewarding ranks above their target.
Excess-only gains receive a penalty capped at 0.25 per transition; incidental
excess accompanying useful progress is not penalized. There is no generic step
bonus. Kills receive a small bonus only with measured requirement progress.
Unknown/mismatched requirement data earns no progress reward. Circle attainment
earns its bonus on the crossing transition only. These weights are initial
shaping choices, not benchmark-validated improvements. Existing running pilots
retain their loaded scoring code until restarted; use a separate experiment
checkpoint to compare versions without mixing their reward histories.

Reward version `balanced-requirements-exp/3` additionally tracks total absorbed
EXP (including completed rank thresholds) and currently pooled EXP separately.
Useful absorbed EXP adds at most 0.1 reward per transition, weighted toward
unmet eligible requirements; above-target skills receive no EXP bonus. Pool
growth itself is telemetry, not a second reward. This keeps rank rollover from
looking like EXP loss and avoids counting ordinary pool-to-rank conversion
twice. Rested conversion bonuses remain part of actual absorbed EXP. The API
and pilot must both run this version to report these fields; missing EXP is
unavailable, not zero.

`POST /api/observe` with `{"text":"..."}` parses a screen event. `GET /api/state` shows the observation and proposal. `POST /api/approve` sends only an unexpired typed allowlisted action through the configured transport. `POST /api/stop` permanently disarms the process. Restart to re-arm. Raw text is never persisted.

The package can be removed with `rm -r dr_adapter`; existing NotFlyGuy modules and state are independent.
