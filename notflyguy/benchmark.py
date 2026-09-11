"""Physical FlyBody motor baselines; not a connectome and not a trained policy.

Naming is explicit and deliberate:

* ``zero``    -- every actuator held at its **minimum**, 0.0. This is the strong
  do-nothing local optimum on ``walk_on_ball`` (mean reward ~0.166 over 200
  steps). NOTE: on the installed FlyBody the live action spec is not [0, 1] for
  every actuator, but the project treats 0.0 as "minimum" and the measured
  constant-action table was produced with an all-zeros array, so ``zero`` means
  literally ``np.zeros``.
* ``neutral`` -- every actuator held at 0.5, the untrained readout's constant
  output (``to_action(0)``). This is *not* mid-range for leg actuators whose
  joint ranges straddle zero; it measures ~0.0 reward.
* ``max``     -- every actuator held at its live ``action_spec().maximum``.
* ``random``  -- uniform samples over the live action spec, one per step.

Output stays JSON-per-line so it can be piped into the dashboard tooling.
"""
import argparse
import json

import numpy as np

from .motor_training import ACTION_MIN, ACTION_NEUTRAL

POLICIES = ('zero', 'neutral', 'max', 'random')


def action_for(policy, spec, rng):
    """Return the action for one step of ``policy``; constants are documented above."""
    if policy == 'zero':
        return np.full(spec.shape, ACTION_MIN, dtype=float)
    if policy == 'neutral':
        return np.full(spec.shape, ACTION_NEUTRAL, dtype=float)
    if policy == 'max':
        return np.asarray(spec.maximum, dtype=float).copy()
    if policy == 'random':
        return rng.uniform(spec.minimum, spec.maximum)
    raise ValueError(f'Unknown policy {policy!r}; expected one of {POLICIES}')


def evaluate(policy, seed, steps):
    # Imported here so importing this module does not build MuJoCo.
    from flybody.fly_envs import walk_on_ball

    env = walk_on_ball(random_state=np.random.RandomState(seed))
    rng = np.random.default_rng(seed)
    timestep = env.reset()
    spec = env.action_spec()
    rewards = []
    speeds = []
    for _ in range(steps):
        action = action_for(policy, spec, rng)
        timestep = env.step(action)
        rewards.append(float(timestep.reward))
        speeds.append(np.asarray(env.physics.named.data.qvel['ball']).copy())
        if timestep.last():
            break
    # Mean over the steps actually executed; the requested count is recorded
    # separately so a truncated episode cannot silently inflate or deflate it.
    executed = len(rewards)
    return {'policy': policy, 'seed': seed, 'steps': executed, 'requested_steps': steps,
            'mean_reward': float(np.mean(rewards)) if executed else 0.0,
            'mean_ball_velocity': np.mean(speeds, axis=0).tolist() if speeds else None,
            'terminated': bool(timestep.last()), 'simulation_seconds': float(env.physics.data.time)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=2,
                        help='evaluate seeds 0..seeds-1')
    parser.add_argument('--policy', choices=POLICIES, action='append',
                        help='policy to run (repeatable); defaults to zero, neutral, random')
    args = parser.parse_args()
    if args.steps < 1 or args.seeds < 1:
        parser.error('steps and seeds must be positive')
    policies = tuple(args.policy) if args.policy else ('zero', 'neutral', 'random')
    for seed in range(args.seeds):
        for policy in policies:
            print(json.dumps(evaluate(policy, seed, args.steps)), flush=True)


if __name__ == '__main__':
    main()
