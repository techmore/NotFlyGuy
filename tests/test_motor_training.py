"""Fast tests for the motor training plumbing.

These tests never build MuJoCo: the readout, the checkpoint round-trip, the
episode accounting, the optimiser, and the baseline comparison are all pure
numpy / duck-typed. Rollouts use a stub environment capped at a few steps.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from notflyguy import motor_training as mt
from notflyguy import train as train_module

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Parameter plumbing ----------------------------------------------------
def test_flat_parameter_vector_is_3107():
    readout = mt.make_readout(seed=0)
    assert int(readout.mask.sum()) == mt.N_WEIGHT_PARAMETERS == 3048
    assert readout.bias.shape == (mt.N_ACTUATORS,)
    assert readout.parameters().shape == (mt.N_PARAMETERS,) == (3107,)


def test_parameter_round_trip():
    readout = mt.make_readout(seed=0)
    rng = np.random.default_rng(4)
    flat = rng.normal(0.0, 0.5, mt.N_PARAMETERS)
    readout.set_parameters(flat)
    assert np.allclose(readout.parameters(), flat)
    # Weights outside the mask must stay zero, and the bias must be the tail.
    assert np.all(readout.weights[~readout.mask] == 0.0)
    assert np.allclose(readout.bias, flat[mt.N_WEIGHT_PARAMETERS:])


def test_set_parameters_rejects_wrong_length():
    readout = mt.make_readout(seed=0)
    with pytest.raises(ValueError):
        readout.set_parameters(np.zeros(mt.N_PARAMETERS - 1))
    with pytest.raises(ValueError):
        readout.set_parameters(np.full(mt.N_PARAMETERS, np.inf))


def test_parameter_space_modes_expand_to_3107():
    readout = mt.make_readout(seed=0)
    bias_space = mt.ParameterSpace(readout, mode='bias')
    full_space = mt.ParameterSpace(readout, mode='full')
    rank2 = mt.ParameterSpace(readout, mode='lowrank', rank=2)
    assert (bias_space.dim, full_space.dim, rank2.dim) == (59, 3107, 59 + 2 * (59 + 381))

    x = np.linspace(-1.0, 1.0, bias_space.dim)
    full = bias_space.to_full(x)
    assert full.shape == (3107,)
    assert np.all(full[:3048] == 0.0)          # bias-only mode leaves weights at zero
    assert np.allclose(full[3048:], x)

    xr = np.linspace(-0.1, 0.1, rank2.dim)
    full_r = rank2.to_full(xr)
    readout.set_parameters(full_r)
    u = xr[59:59 + 59 * 2].reshape(59, 2)
    v = xr[59 + 59 * 2:].reshape(381, 2)
    assert np.allclose(readout.weights[readout.mask], (u @ v.T)[readout.mask])
    assert np.allclose(readout.parameters(), full_r)

    with pytest.raises(ValueError):
        mt.ParameterSpace(readout, mode='nonsense')


def test_parameter_space_zero_init_saturates_near_minimum():
    readout = mt.make_readout(seed=0)
    space = mt.ParameterSpace(readout, mode='bias')
    rng = np.random.default_rng(0)
    x = space.initial('zero', rng)
    readout.set_parameters(space.to_full(x))
    action = readout(np.zeros(mt.N_NEURONS))
    assert np.all(action < 0.01)                # essentially the all-zeros action
    neutral = space.initial('neutral', rng)
    assert np.allclose(space.to_full(neutral)[3048:], 0.0)


# --- Checkpoint round-trip -------------------------------------------------
def test_checkpoint_round_trip_reproduces_identical_actions(tmp_path):
    readout = mt.make_readout(seed=1)
    rng = np.random.default_rng(7)
    flat = rng.normal(0.0, 0.3, mt.N_PARAMETERS)
    readout.set_parameters(flat)
    activity = rng.normal(0.0, 1.0, mt.N_NEURONS)
    before = readout(activity)

    path = tmp_path / 'checkpoint.npz'
    np.savez_compressed(path, parameters=flat)
    restored = mt.make_readout(seed=99)        # different seed: only the file matters
    loaded = mt.load_checkpoint(path, restored)

    assert loaded.shape == (3107,)
    assert np.allclose(loaded, flat)
    assert np.array_equal(before, restored(activity))


# --- Episode accounting (the scoring bug) ----------------------------------
class _Step:
    def __init__(self, reward, last):
        self.reward = reward
        self._last = last

    def last(self):
        return self._last


class _StubEnv:
    """Deterministic stub: rewards are returned in order, then the episode ends."""

    def __init__(self, rewards, terminate_after):
        self.rewards = list(rewards)
        self.terminate_after = terminate_after
        self.index = 0
        self.actions = []

    def reset(self):
        self.index = 0
        self.actions = []
        return _Step(0.0, False)

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float).copy())
        reward = self.rewards[self.index]
        self.index += 1
        return _Step(reward, self.index >= self.terminate_after)


class _ConstantPolicy:
    def __init__(self, value=0.0):
        self.value = value
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, env):
        return np.full(59, self.value)


def test_episode_mean_divides_by_steps_actually_taken():
    env = _StubEnv([1.0] * 5, terminate_after=3)
    policy = _ConstantPolicy()
    result = mt.run_episode(env, policy, seed=0, steps=5, collect_velocity=False)
    assert result.steps == 3
    assert result.requested_steps == 5
    assert result.total_reward == pytest.approx(3.0)
    # The buggy version returned 3.0 / 5 == 0.6 here.
    assert result.mean_reward == pytest.approx(1.0)
    assert result.terminated is True
    assert policy.resets == 1


def test_episode_full_length_mean_and_no_termination():
    env = _StubEnv([1.0, 0.0, 0.5, 0.5], terminate_after=99)
    result = mt.run_episode(env, _ConstantPolicy(), seed=10, steps=4, collect_velocity=False)
    assert (result.steps, result.terminated) == (4, False)
    assert result.mean_reward == pytest.approx(np.mean([1.0, 0.0, 0.5, 0.5]))
    with pytest.raises(ValueError):
        mt.run_episode(env, _ConstantPolicy(), seed=0, steps=0, collect_velocity=False)


def test_constant_policy_emits_documented_values():
    policy = mt.ConstantPolicy((59,), mt.ACTION_MIN)
    assert np.all(policy(None) == 0.0)
    neutral = mt.ConstantPolicy((59,), mt.ACTION_NEUTRAL)
    assert np.all(neutral(None) == 0.5)
    assert mt.ACTION_MIN == 0.0 and mt.ACTION_NEUTRAL == 0.5


# --- Baseline comparison ---------------------------------------------------
def test_compare_to_baselines_reports_per_seed_and_flags():
    candidate = {10: 0.20, 11: 0.18}
    zero = {10: 0.166, 11: 0.166}
    neutral = {10: 0.0, 11: 0.0}
    comparison = mt.compare_to_baselines(candidate, zero, neutral)

    assert comparison['per_seed']['10']['delta_vs_zero'] == pytest.approx(0.034)
    assert comparison['per_seed']['11']['delta_vs_zero'] == pytest.approx(0.014)
    assert comparison['mean_delta_vs_zero'] == pytest.approx(0.024)
    assert comparison['min_delta_vs_zero'] == pytest.approx(0.014)
    assert comparison['beats_zero_baseline'] is True
    assert comparison['beats_zero_on_every_seed'] is True
    assert comparison['beats_zero_by_margin_0.01'] is True
    assert comparison['beats_zero_by_margin_0.05'] is False
    assert comparison['mean_delta_vs_neutral'] == pytest.approx(0.19)
    assert comparison['zero_baseline_per_seed'] == {'10': 0.166, '11': 0.166}
    assert comparison['neutral_baseline_per_seed'] == {'10': 0.0, '11': 0.0}
    assert comparison['candidate_per_seed'] == {'10': 0.20, '11': 0.18}


def test_compare_to_baselines_losing_candidate_is_flagged_false():
    comparison = mt.compare_to_baselines({10: 0.160}, {10: 0.166}, {10: 0.0})
    assert comparison['beats_zero_baseline'] is False
    assert comparison['beats_zero_on_every_seed'] is False
    assert comparison['mean_delta_vs_zero'] == pytest.approx(-0.006)


def test_compare_to_baselines_requires_matching_seeds():
    with pytest.raises(ValueError):
        mt.compare_to_baselines({10: 0.2}, {11: 0.166}, {10: 0.0})
    with pytest.raises(ValueError):
        mt.compare_to_baselines({}, {}, {})


# --- Optimizer -------------------------------------------------------------
def test_cem_improves_a_smooth_objective():
    target = np.array([1.0, -2.0, 0.5, 3.0])

    def objective(x):
        return -float(np.sum((x - target) ** 2))

    config = mt.CEMConfig(population=16, elite_fraction=0.25, iterations=25, seed=3)
    result = mt.cem(objective, np.zeros(4), np.ones(4), config)
    assert result['best_score'] > -0.5
    assert len(result['history']) == 25
    assert result['history'][-1]['iteration'] == 25
    assert result['best'].shape == (4,)


def test_cem_config_validation():
    with pytest.raises(ValueError):
        mt.CEMConfig(population=1)
    with pytest.raises(ValueError):
        mt.CEMConfig(elite_fraction=0.0)
    with pytest.raises(ValueError):
        mt.CEMConfig(population=4, elite_fraction=1.0)


# --- CLI validation and import isolation -----------------------------------
def test_parse_seeds_and_cli_validation():
    assert mt.parse_seeds('0,1,2') == (0, 1, 2)
    assert mt.parse_seeds(' 10 , 11 ') == (10, 11)
    for bad in ('', 'a', '1,1', '-1'):
        with pytest.raises(ValueError):
            mt.parse_seeds(bad)

    parser = train_module.build_parser()
    args = parser.parse_args(['--train-seeds', '0,1', '--eval-seeds', '10'])
    assert train_module.validate_args(args) == ((0, 1), (10,))
    # Overlapping train/eval seeds must be refused.
    overlapping = parser.parse_args(['--train-seeds', '0,1', '--eval-seeds', '1'])
    with pytest.raises(SystemExit):
        train_module.validate_args(overlapping)
    for bad in (['--steps', '0'], ['--population', '1'], ['--elite-fraction', '0'],
                ['--rank', '0'], ['--init-bias-std', '0']):
        with pytest.raises(SystemExit):
            train_module.validate_args(parser.parse_args(bad))


def test_train_module_imports_without_mujoco_or_flybody():
    code = (
        'import sys; import notflyguy.train; '
        "assert 'flybody' not in sys.modules, 'importing notflyguy.train pulled in flybody'; "
        "assert 'mujoco' not in sys.modules, 'importing notflyguy.train pulled in mujoco'; "
        "print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'ok' in completed.stdout


def test_benchmark_module_imports_without_mujoco():
    code = (
        'import sys; import notflyguy.benchmark; '
        "assert 'flybody' not in sys.modules; print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'ok' in completed.stdout
