"""Bounded FlyBody motor baselines; not a connectome or a trained policy."""
import argparse
import json
import numpy as np
from flybody.fly_envs import walk_on_ball


def evaluate(policy, seed, steps):
    env = walk_on_ball(random_state=np.random.RandomState(seed))
    rng = np.random.default_rng(seed)
    timestep = env.reset()
    spec = env.action_spec()
    rewards = []
    speeds = []
    for _ in range(steps):
        action = np.zeros(spec.shape) if policy == 'zero' else rng.uniform(spec.minimum, spec.maximum)
        timestep = env.step(action)
        rewards.append(float(timestep.reward))
        speeds.append(np.asarray(env.physics.named.data.qvel['ball']).copy())
        if timestep.last():
            break
    return {'policy': policy, 'seed': seed, 'steps': len(rewards),
            'mean_reward': float(np.mean(rewards)),
            'mean_ball_velocity': np.mean(speeds, axis=0).tolist(),
            'terminated': timestep.last(), 'simulation_seconds': float(env.physics.data.time)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.seeds < 1:
        parser.error('steps and seeds must be positive')
    for seed in range(args.seeds):
        for policy in ('zero', 'random'):
            print(json.dumps(evaluate(policy, seed, args.steps)), flush=True)


if __name__ == '__main__':
    main()
