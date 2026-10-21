"""Honest trainer for the modeled leg-motor readout.

The script searches a readout from measured motor-neuron activity to the 59
FlyBody actuators with a real optimizer (diagonal CEM) over a reduced or full
parameterization, then reports held-out results against two physical baselines
measured with the *same* seeds and the *same* step counts:

* ``zero``    -- every actuator commanded to its minimum (0.0),
* ``neutral`` -- the untrained readout's constant 0.5.

It is designed to be able to fail loudly. If the candidate does not beat the
zero baseline on held-out seeds, ``report.json`` says so and
``beats_zero_baseline`` is ``false``. Nothing here is a claim about a real fly:
the connectome supplies structure and the readout is an explicit model.

Accounting fix: mean reward is ``total_reward / steps_actually_taken``. The
earlier version divided by the requested step count while breaking early on
``timestep.last()``, which understated truncated episodes.

MuJoCo/FlyBody is imported inside :func:`main`, so importing this module (for
tests or tooling) does not build a physics simulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import motor_training as mt
from .circuit import MotorCircuitController
from .motor_projection import limits_from_spec

DEFAULT_OUTPUT = 'data/training/readout-cem'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train the LegMotorReadout against held-out baselines.',
    )
    parser.add_argument('--steps', type=int, default=200,
                        help='episode length used for training rollouts')
    parser.add_argument('--eval-steps', type=int, default=1000,
                        help='episode length used for held-out evaluation (1000 = full episode)')
    parser.add_argument('--iterations', type=int, default=20, help='CEM iterations')
    parser.add_argument('--population', type=int, default=16, help='CEM samples per iteration')
    parser.add_argument('--elite-fraction', type=float, default=0.25,
                        help='fraction of each CEM population kept as elites')
    parser.add_argument('--parameterization', choices=('bias', 'lowrank', 'full'), default='bias',
                        help='bias = 59 tonic biases only; lowrank = biases + rank-r weight update; '
                             'full = all 3107 parameters')
    parser.add_argument('--rank', type=int, default=2, help='rank for --parameterization lowrank')
    parser.add_argument('--init', choices=('zero', 'neutral', 'random'), default='neutral',
                        help='starting point: neutral is the untrained 0.5 readout (maximum tanh '
                             'sensitivity); zero saturates near the actuator minimum, i.e. the '
                             'measured all-zeros optimum, where the readout has almost no authority')
    parser.add_argument('--init-bias-std', type=float, default=1.0,
                        help='initial CEM std for bias dimensions (drive/logit units). Around 1.0 '
                             'the first samples cover most of the reachable (0, 1) action range.')
    parser.add_argument('--init-weight-std', type=float, default=0.1,
                        help='initial CEM std for weight dimensions')
    parser.add_argument('--std-smoothing', type=float, default=0.5,
                        help='blend of the previous CEM std into the elite std (0 collapses fastest, '
                             'higher values keep exploration alive)')
    parser.add_argument('--train-seeds', default='0,1,2', help='comma-separated training seeds')
    parser.add_argument('--eval-seeds', default='10,11,12,13', help='comma-separated held-out seeds')
    parser.add_argument('--seeds-per-sample', type=int, default=1,
                        help='training seeds averaged per CEM sample, cycled through --train-seeds '
                             '(0 = all training seeds). The sampler still covers every training seed.')
    parser.add_argument('--seed', type=int, default=17, help='optimizer RNG seed')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='new output directory')
    parser.add_argument('--catalog-root', default=None,
                        help='directory holding leg-motor-catalog.json (default: '
                             f'NOTFLYGUY_DATA_ROOT or {mt.DEFAULT_CATALOG_ROOT})')
    return parser


def validate_args(args: argparse.Namespace) -> tuple:
    """Validate CLI arguments; return (train_seeds, eval_seeds)."""
    if args.steps < 1:
        raise SystemExit('--steps must be positive')
    if args.eval_steps < 1:
        raise SystemExit('--eval-steps must be positive')
    if args.iterations < 1:
        raise SystemExit('--iterations must be positive')
    if args.population < 2:
        raise SystemExit('--population must be at least 2')
    if not 0.0 < args.elite_fraction <= 1.0:
        raise SystemExit('--elite-fraction must be in (0, 1]')
    if max(1, int(round(args.population * args.elite_fraction))) >= args.population:
        raise SystemExit('--elite-fraction must leave at least one non-elite sample')
    if args.rank < 1:
        raise SystemExit('--rank must be positive')
    if args.seeds_per_sample < 0:
        raise SystemExit('--seeds-per-sample must be non-negative (0 = all training seeds)')
    if args.init_bias_std <= 0.0 or args.init_weight_std <= 0.0:
        raise SystemExit('initial standard deviations must be positive')
    if not 0.0 <= args.std_smoothing < 1.0:
        raise SystemExit('--std-smoothing must be in [0, 1)')
    try:
        train_seeds = mt.parse_seeds(args.train_seeds)
        eval_seeds = mt.parse_seeds(args.eval_seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    overlap = sorted(set(train_seeds) & set(eval_seeds))
    if overlap:
        raise SystemExit(
            f'Training and evaluation seeds must be disjoint; shared seeds: {overlap}'
        )
    return train_seeds, eval_seeds


def _actuator_names(env) -> tuple:
    """Read actuator names from the live environment."""
    names = env.physics.named.model.actuator_ctrlrange.axes.row.names
    return tuple(str(name) for name in names)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train_seeds, eval_seeds = validate_args(args)

    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f'{output} already exists; choose a new output directory to preserve prior runs'
        )

    # Imported here (not at module import) so that importing notflyguy.train
    # never constructs MuJoCo.
    from flybody.fly_envs import walk_on_ball

    started = {'train_seeds': train_seeds, 'eval_seeds': eval_seeds}
    print(json.dumps({'stage': 'config', **started, 'steps': args.steps,
                      'eval_steps': args.eval_steps,
                      'parameterization': args.parameterization,
                      'population': args.population, 'iterations': args.iterations}), flush=True)

    # One environment per seed, reused by reset(). Episodes are deterministic
    # and, as recorded in the report, are in fact identical across seeds for
    # this task build.
    envs = {}
    for seed in sorted(set(train_seeds) | set(eval_seeds)):
        envs[seed] = walk_on_ball(random_state=np.random.RandomState(seed))
    any_env = envs[train_seeds[0]]
    spec = any_env.action_spec()
    actuator_names = _actuator_names(any_env)
    if len(actuator_names) != mt.N_ACTUATORS:
        raise SystemExit(f'Expected {mt.N_ACTUATORS} actuators, environment has {len(actuator_names)}')
    names_match_constant = actuator_names == mt.DEFAULT_ACTUATOR_NAMES
    if not names_match_constant:
        print(json.dumps({'stage': 'warning',
                          'message': 'live actuator names differ from the captured constant; '
                                     'using the live order'}), flush=True)

    controller = MotorCircuitController(
        args.catalog_root, actuator_names,
        action_limits=limits_from_spec(spec))
    space = mt.ParameterSpace(controller.readout, mode=args.parameterization, rank=args.rank)

    config = mt.CEMConfig(
        population=args.population,
        elite_fraction=args.elite_fraction,
        iterations=args.iterations,
        bias_std=args.init_bias_std,
        weight_std=args.init_weight_std,
        std_smoothing=args.std_smoothing,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    mean0 = space.initial(args.init, rng, bias_std=args.init_bias_std,
                          weight_std=args.init_weight_std)
    std0 = space.per_dimension_std(args.init_bias_std, args.init_weight_std)

    readout_policy = mt.ReadoutPolicy(controller)
    zero_policy = mt.ConstantPolicy(spec.shape, mt.ACTION_MIN)
    neutral_policy = mt.ConstantPolicy(spec.shape, mt.ACTION_NEUTRAL)

    def score(parameters_full: np.ndarray, seeds, steps: int) -> dict:
        controller.readout.set_parameters(parameters_full)
        return {seed: mt.run_episode(envs[seed], readout_policy, seed, steps)
                for seed in seeds}

    # Each CEM sample is scored on a rotating subset of the training seeds so
    # that a limited rollout budget still touches every training seed instead of
    # paying for all of them on every sample.
    sampling_state = {'cursor': 0}

    def training_seeds_for_sample():
        width = len(train_seeds) if args.seeds_per_sample == 0 else args.seeds_per_sample
        width = max(1, min(width, len(train_seeds)))
        start = sampling_state['cursor'] % len(train_seeds)
        sampling_state['cursor'] += 1
        return tuple(train_seeds[(start + i) % len(train_seeds)] for i in range(width))

    def objective(x: np.ndarray) -> float:
        replies = score(space.to_full(x), training_seeds_for_sample(), args.steps)
        return float(np.mean([r.mean_reward for r in replies.values()]))

    def report_iteration(record: dict) -> None:
        print(json.dumps({'stage': 'cem', **record}), flush=True)

    initial_score = objective(mean0)
    print(json.dumps({'stage': 'initial', 'score': initial_score}), flush=True)

    result = mt.cem(objective, mean0, std0, config, on_iteration=report_iteration)
    elapsed_history = result['history']

    # Deployable candidate = elite mean; honest best = best single sample seen.
    candidates = {
        'cem_mean': space.to_full(result['mean']),
        'best_sample': space.to_full(result['best']),
    }

    # ---------------------------------------------------------------- eval
    def evaluate_policy(policy, seeds, steps: int) -> dict:
        return {seed: mt.run_episode(envs[seed], policy, seed, steps)
                for seed in seeds}

    zero_eval = evaluate_policy(zero_policy, eval_seeds, args.eval_steps)
    neutral_eval = evaluate_policy(neutral_policy, eval_seeds, args.eval_steps)

    comparisons = {}
    for name, parameters in candidates.items():
        candidate_eval = score(parameters, eval_seeds, args.eval_steps)
        comparison = mt.compare_to_baselines(
            mt.mean_rewards(candidate_eval),
            mt.mean_rewards(zero_eval),
            mt.mean_rewards(neutral_eval),
        )
        comparison['candidate_total_rewards'] = {
            str(s): candidate_eval[s].total_reward for s in sorted(candidate_eval)
        }
        comparison['candidate_steps_per_seed'] = {
            str(s): candidate_eval[s].steps for s in sorted(candidate_eval)
        }
        comparisons[name] = comparison

    primary = comparisons['cem_mean']
    zero_scores = [mt.mean_rewards(zero_eval)[s] for s in eval_seeds]
    seeds_degenerate = bool(np.allclose(zero_scores, zero_scores[0], rtol=0.0, atol=0.0))

    # -------------------------------------------------------- checkpoint
    output.mkdir(parents=True)
    np.savez_compressed(
        output / 'checkpoint.npz',
        parameters=candidates['cem_mean'],
        parameterization=np.array(args.parameterization),
        rank=np.array(args.rank),
        steps=np.array(args.steps),
        eval_steps=np.array(args.eval_steps),
        train_seeds=np.array(train_seeds),
        eval_seeds=np.array(eval_seeds),
        actuator_names=np.array(actuator_names),
    )
    np.savez_compressed(
        output / 'checkpoint_best_sample.npz',
        parameters=candidates['best_sample'],
        parameterization=np.array(args.parameterization),
        rank=np.array(args.rank),
    )

    report = {
        'algorithm': 'diagonal cross-entropy method (CEM) over a reduced readout parameterization',
        'optimizer': {
            'name': 'diagonal CEM',
            'population': args.population,
            'elite_count': config.elite_count,
            'elite_fraction': args.elite_fraction,
            'iterations': args.iterations,
            'rng_seed': args.seed,
            'std_smoothing': args.std_smoothing,
            'seeds_per_sample': args.seeds_per_sample,
            'objective': ('mean per-step reward over a rotating subset of the training seeds, at '
                          '--steps per episode; 0 seeds-per-sample means all training seeds each time'),
        },
        'parameterization': {
            'mode': args.parameterization,
            'rank': args.rank,
            'dim': int(space.dim),
            'full_dim': int(space.n_full),
            'init': args.init,
            'init_bias_std': args.init_bias_std,
            'init_weight_std': args.init_weight_std,
            'note': ('Optimizer search space is reduced; the deployed artifact is always the full '
                     'flat 3107-parameter readout vector from LegMotorReadout.parameters().'),
        },
        'steps_per_episode': {'train': args.steps, 'eval': args.eval_steps},
        'train_seeds': list(train_seeds),
        'eval_seeds': list(eval_seeds),
        'zero_baseline_per_seed': primary['zero_baseline_per_seed'],
        'neutral_baseline_per_seed': primary['neutral_baseline_per_seed'],
        'candidate_per_seed': primary['candidate_per_seed'],
        'delta_vs_zero_per_seed': primary['delta_vs_zero_per_seed'],
        'delta_vs_neutral_per_seed': primary['delta_vs_neutral_per_seed'],
        'mean_delta_vs_zero': primary['mean_delta_vs_zero'],
        'mean_delta_vs_neutral': primary['mean_delta_vs_neutral'],
        'beats_zero_baseline': primary['beats_zero_baseline'],
        'beats_zero_on_every_seed': primary['beats_zero_on_every_seed'],
        'beats_zero_by_margin_0.01': primary['beats_zero_by_margin_0.01'],
        'beats_zero_by_margin_0.05': primary['beats_zero_by_margin_0.05'],
        'mean_candidate': primary['mean_candidate'],
        'mean_zero_baseline': primary['mean_zero_baseline'],
        'mean_neutral_baseline': primary['mean_neutral_baseline'],
        'min_delta_vs_zero': primary['min_delta_vs_zero'],
        'comparisons': comparisons,
        'training': {
            'initial_score_at_init_point': initial_score,
            'best_sample_score_in_training': result['best_score'],
            'history': elapsed_history,
        },
        'artifacts': {
            'checkpoint': 'checkpoint.npz (parameters = flat 3107 vector for set_parameters)',
            'checkpoint_best_sample': 'checkpoint_best_sample.npz',
        },
        'environment': {
            'actuator_names_match_captured_constant': bool(names_match_constant),
            'action_spec_min_unique': [float(v) for v in np.unique(spec.minimum)],
            'action_spec_max_unique': [float(v) for v in np.unique(spec.maximum)],
            'note': ('The live FlyBody action spec is NOT [0, 1] for every actuator: head/abdomen '
                     'actuators span [0, 1] while leg actuators span their joint ranges (e.g. '
                     'coxa_T1_left [-3, 3]). The readout is an affine map onto each actuator\'s '
                     'declared range from this spec, so the full per-actuator range is reachable '
                     'and a zero drive lands on that actuator\'s own midpoint.'),
            'zero_baseline_is_seed_degenerate': seeds_degenerate,
            'seed_degeneracy_note': (
                'walk_on_ball() does not randomize the episode: every seed produced a bit-identical '
                'zero-baseline reward in this build, so "held-out seeds" are nominal only. The '
                f'genuine held-out axis here is episode length (train {args.steps} steps vs eval '
                f'{args.eval_steps} steps).'
            ) if seeds_degenerate else 'Held-out seeds produced distinct baseline rewards.',
        },
        'limitations': [
            'The readout is a modeled mapping from measured motor-neuron activity to actuators, '
            'not an identified biological motor program.',
            'Held-out seeds are numerically degenerate in this FlyBody build, so cross-seed '
            'generalization is not actually tested; only length extrapolation to the full episode.',
            'The readout is an affine map onto the live per-actuator action ranges, so no part of '
            'the actuator range is structurally unreachable.',
            'Reward is the product of three linear tolerance factors; any lateral ball velocity '
            'penalises the score multiplicatively, so a coordinated gait rather than a held '
            'posture is required to approach a reward of 1.0.',
            'The checkpoint is not deployed automatically.',
        ],
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=False) + '\n')

    summary = {
        'stage': 'summary',
        'zero_baseline': primary['mean_zero_baseline'],
        'neutral_baseline': primary['mean_neutral_baseline'],
        'candidate': primary['mean_candidate'],
        'mean_delta_vs_zero': primary['mean_delta_vs_zero'],
        'beats_zero_baseline': primary['beats_zero_baseline'],
        'output': str(output),
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
