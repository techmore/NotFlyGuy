"""Shared training machinery for the modeled leg-motor readout.

This module deliberately contains **no** MuJoCo or FlyBody import. The
environment is created by the caller and passed in as a duck-typed object, so
the parameter plumbing, the episode accounting, the optimiser, and the
baseline comparison can all be exercised in fast unit tests without paying for
(or depending on) a physics build.

Honest scope note: everything here optimises an explicitly *modeled* readout
from measured motor-neuron activity to FlyBody actuators. Nothing here claims a
biological motor program.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import CATALOG_FILE, DEFAULT_DATA_ROOT, data_root

# --- Frozen dimensions -----------------------------------------------------
# The catalog holds 381 leg motor neurons and ``walk_on_ball`` exposes 59
# actuators, so the mask has 3048 selectable weights plus 59 biases.
N_ACTUATORS = 59
N_NEURONS = 381
N_WEIGHT_PARAMETERS = 3048
N_PARAMETERS = N_WEIGHT_PARAMETERS + N_ACTUATORS  # 3107

# Neutral / minimum action, mirroring ``motor_projection`` without importing
# the physics-facing constants dynamically.
ACTION_MIN = 0.0
ACTION_NEUTRAL = 0.5

# Actuator order captured from ``walk_on_ball().action_spec()`` on the pinned
# FlyBody commit. ``train.py`` re-reads the live names and refuses to proceed on
# a mismatch, so this constant cannot silently drift from the environment.
DEFAULT_ACTUATOR_NAMES = (
    'walker/head_abduct', 'walker/head_twist', 'walker/head',
    'walker/abdomen_abduct', 'walker/abdomen', 'walker/coxa_abduct_T1_left',
    'walker/coxa_twist_T1_left', 'walker/coxa_T1_left', 'walker/femur_twist_T1_left',
    'walker/femur_T1_left', 'walker/tibia_T1_left', 'walker/tarsus_T1_left',
    'walker/tarsus2_T1_left', 'walker/coxa_abduct_T1_right', 'walker/coxa_twist_T1_right',
    'walker/coxa_T1_right', 'walker/femur_twist_T1_right', 'walker/femur_T1_right',
    'walker/tibia_T1_right', 'walker/tarsus_T1_right', 'walker/tarsus2_T1_right',
    'walker/coxa_abduct_T2_left', 'walker/coxa_twist_T2_left', 'walker/coxa_T2_left',
    'walker/femur_twist_T2_left', 'walker/femur_T2_left', 'walker/tibia_T2_left',
    'walker/tarsus_T2_left', 'walker/tarsus2_T2_left', 'walker/coxa_abduct_T2_right',
    'walker/coxa_twist_T2_right', 'walker/coxa_T2_right', 'walker/femur_twist_T2_right',
    'walker/femur_T2_right', 'walker/tibia_T2_right', 'walker/tarsus_T2_right',
    'walker/tarsus2_T2_right', 'walker/coxa_abduct_T3_left', 'walker/coxa_twist_T3_left',
    'walker/coxa_T3_left', 'walker/femur_twist_T3_left', 'walker/femur_T3_left',
    'walker/tibia_T3_left', 'walker/tarsus_T3_left', 'walker/tarsus2_T3_left',
    'walker/coxa_abduct_T3_right', 'walker/coxa_twist_T3_right', 'walker/coxa_T3_right',
    'walker/femur_twist_T3_right', 'walker/femur_T3_right', 'walker/tibia_T3_right',
    'walker/tarsus_T3_right', 'walker/tarsus2_T3_right', 'walker/adhere_claw_T1_left',
    'walker/adhere_claw_T1_right', 'walker/adhere_claw_T2_left', 'walker/adhere_claw_T2_right',
    'walker/adhere_claw_T3_left', 'walker/adhere_claw_T3_right',
)

DEFAULT_CATALOG_ROOT = DEFAULT_DATA_ROOT


# --- Readout construction --------------------------------------------------
def make_readout(root=None, actuator_names=None, seed=0, limits=None):
    """Build the frozen :class:`LegMotorReadout` without a MuJoCo import.

    ``root`` defaults to :func:`notflyguy.paths.data_root` (honouring
    ``NOTFLYGUY_DATA_ROOT``). ``limits`` is the per-actuator ``[min, max]``
    from the live action spec. Omitting it falls back to a single ``(0, 1)``
    range, which is *not* the live FlyBody spec; pass
    ``limits_from_spec(env.action_spec())`` to search the space the body
    actually accepts.
    """
    from .motor_projection import LegMotorReadout

    catalog = data_root(root) / CATALOG_FILE
    if not catalog.exists():
        raise FileNotFoundError(
            f'Missing {catalog}. Run scripts/catalog_leg_neurons.py first, or point '
            'NOTFLYGUY_DATA_ROOT at a directory that holds it.'
        )
    neurons = json.loads(catalog.read_text())['neurons']
    names = tuple(actuator_names) if actuator_names is not None else DEFAULT_ACTUATOR_NAMES
    if len(names) != N_ACTUATORS:
        raise ValueError(f'Expected {N_ACTUATORS} actuator names, got {len(names)}')
    return LegMotorReadout(neurons, names, seed=seed, limits=limits)


def load_checkpoint(path, readout) -> np.ndarray:
    """Restore a checkpoint written by ``train.py`` into ``readout`` in place."""
    data = np.load(Path(path), allow_pickle=False)
    if 'parameters' not in data:
        raise KeyError('Checkpoint has no "parameters" array')
    parameters = np.asarray(data['parameters'], dtype=float)
    readout.set_parameters(parameters)
    return parameters


# --- Reduced / structured parameterizations --------------------------------
@dataclass
class ParameterSpace:
    """Map a (possibly reduced) optimiser vector onto the flat 3107 vector.

    ``mode``:
      * ``bias``    -- 59 tonic biases only; weights stay zero. The readout then
        emits a constant action vector, which is the cheapest way to probe
        whether a static posture can beat the do-nothing optimum.
      * ``lowrank`` -- biases plus ``U @ V.T`` (rank ``rank``) written only into
        the mask. Adds activity-dependent modulation without 3048 free weights.
      * ``full``    -- all 3107 masked weights and biases.
    """

    readout: object
    mode: str = 'bias'
    rank: int = 2

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.readout.mask, dtype=bool)
        self.n_weights = int(self.mask.sum())
        self.n_neurons = int(self.mask.shape[1])
        self.n_actuators = int(self.mask.shape[0])
        self.n_bias = int(np.asarray(self.readout.bias).size)
        self.n_full = self.n_weights + self.n_bias
        if self.n_full != N_PARAMETERS:
            raise ValueError(
                f'Readout exposes {self.n_full} parameters, expected {N_PARAMETERS}'
            )
        if self.mode == 'full':
            self.dim = self.n_full
        elif self.mode == 'bias':
            self.dim = self.n_bias
        elif self.mode == 'lowrank':
            if self.rank < 1:
                raise ValueError('rank must be positive')
            self.dim = self.n_bias + self.rank * (self.n_actuators + self.n_neurons)
        else:
            raise ValueError(f'Unknown parameterization {self.mode!r}')

    # -- layout helpers -----------------------------------------------------
    def _slices(self):
        if self.mode == 'full':
            return None
        bias = slice(0, self.n_bias)
        if self.mode == 'bias':
            return {'bias': bias}
        u_size = self.n_actuators * self.rank
        return {
            'bias': bias,
            'u': slice(self.n_bias, self.n_bias + u_size),
            'v': slice(self.n_bias + u_size, self.n_bias + u_size + self.n_neurons * self.rank),
        }

    def to_full(self, x: np.ndarray) -> np.ndarray:
        """Expand an optimiser vector into the flat 3107-parameter vector."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f'Expected a vector of length {self.dim}, got {x.shape}')
        if self.mode == 'full':
            return x.copy()
        cut = self._slices()
        weights = np.zeros((self.n_actuators, self.n_neurons), dtype=float)
        if self.mode == 'lowrank':
            u = x[cut['u']].reshape(self.n_actuators, self.rank)
            v = x[cut['v']].reshape(self.n_neurons, self.rank)
            weights = u @ v.T
        return np.concatenate([weights[self.mask], np.asarray(x[cut['bias']], dtype=float)])

    def initial(self, mode: str, rng: np.random.Generator, bias_std: float = 0.5,
                weight_std: float = 0.1, zero_bias: float = -4.0) -> np.ndarray:
        """Starting optimiser vector for ``mode`` in {neutral, zero, random}.

        ``zero`` places the biases where ``to_action`` saturates near the
        actuator minimum (0.0), i.e. it starts at the known local optimum
        measured for the all-zeros action. That makes the search an explicit
        attempt to *escape* that optimum rather than to rediscover it.
        """
        x = np.zeros(self.dim, dtype=float)
        if self.mode != 'full':
            x[self._slices()['bias']] = zero_bias  # applies to zero and random alike
        if mode == 'neutral':
            return np.zeros(self.dim, dtype=float)
        if mode == 'zero':
            return x
        if mode == 'random':
            if self.mode == 'full':
                return rng.normal(0.0, weight_std, self.dim)
            cut = self._slices()
            x = np.zeros(self.dim, dtype=float)
            x[cut['bias']] = rng.normal(0.0, bias_std, self.n_bias)
            if self.mode == 'lowrank':
                x[cut['u']] = rng.normal(0.0, weight_std, cut['u'].stop - cut['u'].start)
                x[cut['v']] = rng.normal(0.0, weight_std, cut['v'].stop - cut['v'].start)
            return x
        raise ValueError(f'Unknown init mode {mode!r}')

    def per_dimension_std(self, bias_std: float, weight_std: float) -> np.ndarray:
        """Initial CEM standard deviation per dimension."""
        if self.mode == 'full':
            return np.full(self.dim, weight_std, dtype=float)
        cut = self._slices()
        std = np.zeros(self.dim, dtype=float)
        std[cut['bias']] = bias_std
        if self.mode == 'lowrank':
            std[cut['u']] = weight_std
            std[cut['v']] = weight_std
        return std


# --- Episode accounting ----------------------------------------------------
@dataclass
class EpisodeResult:
    seed: int
    requested_steps: int
    steps: int
    total_reward: float
    mean_reward: float
    terminated: bool
    mean_ball_velocity: list | None = None

    def as_dict(self) -> dict:
        out = {
            'seed': self.seed,
            'requested_steps': self.requested_steps,
            'steps': self.steps,
            'total_reward': self.total_reward,
            'mean_reward': self.mean_reward,
            'terminated': self.terminated,
        }
        if self.mean_ball_velocity is not None:
            out['mean_ball_velocity'] = list(self.mean_ball_velocity)
        return out


def run_episode(env, policy, seed: int, steps: int, collect_velocity: bool = True) -> EpisodeResult:
    """Roll one episode and average over the steps actually executed.

    The previous trainer divided by the *requested* step count and broke early
    on ``timestep.last()``, which understated every truncated episode. Here the
    denominator is the number of steps really taken.
    """
    if steps < 1:
        raise ValueError('steps must be positive')
    env.reset()
    policy.reset()
    total = 0.0
    executed = 0
    terminated = False
    velocities: list = []
    for _ in range(steps):
        action = policy(env)
        timestep = env.step(action)
        total += float(timestep.reward)
        executed += 1
        if collect_velocity:
            physics = getattr(env, 'physics', None)
            if physics is not None:
                velocities.append(np.asarray(physics.named.data.qvel['ball'], dtype=float).copy())
        if timestep.last():
            terminated = True
            break
    mean_velocity = None
    if velocities:
        mean_velocity = np.mean(velocities, axis=0).tolist()
    return EpisodeResult(
        seed=seed,
        requested_steps=steps,
        steps=executed,
        total_reward=total,
        mean_reward=total / executed if executed else 0.0,
        terminated=terminated,
        mean_ball_velocity=mean_velocity,
    )


# --- Policies --------------------------------------------------------------
class ConstantPolicy:
    """Emit one constant action vector (e.g. all-minimum or all-neutral)."""

    def __init__(self, action_shape, value: float):
        self.action = np.full(tuple(action_shape), float(value), dtype=float)
        self.value = float(value)

    def reset(self) -> None:  # noqa: D401 - part of the policy protocol
        return None

    def __call__(self, env) -> np.ndarray:  # noqa: ARG002 - env unused by design
        return self.action.copy()


class ReadoutPolicy:
    """Drive the measured circuit from ball velocity and read out actuators."""

    def __init__(self, controller):
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def __call__(self, env) -> np.ndarray:
        velocity = np.asarray(env.physics.named.data.qvel['ball'], dtype=float)
        return self.controller.step(velocity)


# --- Optimiser -------------------------------------------------------------
@dataclass
class CEMConfig:
    population: int = 16
    elite_fraction: float = 0.25
    iterations: int = 20
    bias_std: float = 1.0
    weight_std: float = 0.1
    std_floor: float = 1e-3
    # Blend the previous std into the elite std. Without this, a diagonal CEM
    # collapses its variance within a few iterations and strands dimensions
    # that the first few elites happened to agree on -- which matters here
    # because the search starts at the measured all-zeros optimum and must keep
    # enough variance to escape it.
    std_smoothing: float = 0.5
    seed: int = 17

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError('population must be at least 2')
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError('elite_fraction must be in (0, 1]')
        if self.iterations < 1:
            raise ValueError('iterations must be positive')
        if not 0.0 <= self.std_smoothing < 1.0:
            raise ValueError('std_smoothing must be in [0, 1)')
        elite = max(1, int(round(self.population * self.elite_fraction)))
        if elite >= self.population:
            raise ValueError('elite_fraction must leave at least one non-elite sample')
        self.elite_count = elite


def cem(objective: Callable[[np.ndarray], float], mean: np.ndarray, std: np.ndarray,
        config: CEMConfig, on_iteration: Callable[[dict], None] | None = None) -> dict:
    """Diagonal cross-entropy method over a reduced parameter vector.

    Returns the elite mean, the best sample ever evaluated, and a per-iteration
    history. Both the elite mean and the best sample are returned because the
    mean is the more robust deployable choice while the best sample is the
    honest upper bound reached during the search.
    """
    mean = np.asarray(mean, dtype=float).copy()
    std = np.asarray(std, dtype=float).copy()
    if mean.shape != std.shape:
        raise ValueError('mean and std must have the same shape')
    rng = np.random.default_rng(config.seed)
    best_x = mean.copy()
    best_score = float(objective(mean))
    history = []
    for iteration in range(config.iterations):
        samples = mean + std * rng.standard_normal((config.population, mean.size))
        scores = np.array([float(objective(s)) for s in samples], dtype=float)
        order = np.argsort(scores)[::-1]
        elites = samples[order[:config.elite_count]]
        elite_scores = scores[order[:config.elite_count]]
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_x = samples[order[0]].copy()
        mean = elites.mean(axis=0)
        new_std = elites.std(axis=0)
        new_std = (1.0 - config.std_smoothing) * new_std + config.std_smoothing * std
        std = np.maximum(new_std, config.std_floor)
        record = {
            'iteration': iteration + 1,
            'elite_mean_score': float(elite_scores.mean()),
            'best_sample_score': float(scores[order[0]]),
            'best_so_far': float(best_score),
            'std_mean': float(np.mean(std)),
        }
        history.append(record)
        if on_iteration is not None:
            on_iteration(record)
    return {'mean': mean, 'best': best_x, 'best_score': best_score, 'history': history}


# --- Baseline comparison ---------------------------------------------------
def mean_rewards(results: Mapping[int, EpisodeResult]) -> dict:
    return {int(seed): float(result.mean_reward) for seed, result in results.items()}


def compare_to_baselines(candidate: Mapping[int, float], zero: Mapping[int, float],
                         neutral: Mapping[int, float]) -> dict:
    """Per-seed comparison against the two required physical baselines.

    ``zero`` is the all-minimum action (0.0 for every actuator) and ``neutral``
    is the untrained readout's constant 0.5. Raises if the seed sets disagree,
    because a comparison over different seeds would be meaningless.
    """
    seeds = sorted(int(s) for s in candidate)
    if not seeds:
        raise ValueError('No candidate seeds to compare')
    for name, table in (('zero', zero), ('neutral', neutral)):
        if sorted(int(s) for s in table) != seeds:
            raise ValueError(f'Baseline {name!r} must be measured on exactly the candidate seeds')

    per_seed = {}
    for seed in seeds:
        candidate_score = float(candidate[seed])
        zero_score = float(zero[seed])
        neutral_score = float(neutral[seed])
        per_seed[seed] = {
            'candidate': candidate_score,
            'zero': zero_score,
            'neutral': neutral_score,
            'delta_vs_zero': candidate_score - zero_score,
            'delta_vs_neutral': candidate_score - neutral_score,
        }
    deltas_zero = np.array([per_seed[s]['delta_vs_zero'] for s in seeds])
    deltas_neutral = np.array([per_seed[s]['delta_vs_neutral'] for s in seeds])
    candidates = np.array([per_seed[s]['candidate'] for s in seeds])
    zero_scores = np.array([per_seed[s]['zero'] for s in seeds])
    neutral_scores = np.array([per_seed[s]['neutral'] for s in seeds])
    mean_delta_zero = float(deltas_zero.mean())
    return {
        'per_seed': {str(s): per_seed[s] for s in seeds},
        'zero_baseline_per_seed': {str(s): per_seed[s]['zero'] for s in seeds},
        'neutral_baseline_per_seed': {str(s): per_seed[s]['neutral'] for s in seeds},
        'candidate_per_seed': {str(s): per_seed[s]['candidate'] for s in seeds},
        'delta_vs_zero_per_seed': {str(s): per_seed[s]['delta_vs_zero'] for s in seeds},
        'delta_vs_neutral_per_seed': {str(s): per_seed[s]['delta_vs_neutral'] for s in seeds},
        'mean_candidate': float(candidates.mean()),
        'mean_zero_baseline': float(zero_scores.mean()),
        'mean_neutral_baseline': float(neutral_scores.mean()),
        'mean_delta_vs_zero': mean_delta_zero,
        'mean_delta_vs_neutral': float(deltas_neutral.mean()),
        'min_delta_vs_zero': float(deltas_zero.min()),
        'std_delta_vs_zero': float(deltas_zero.std()),
        # Strict improvement of the mean over held-out seeds.
        'beats_zero_baseline': bool(mean_delta_zero > 0.0),
        'beats_zero_on_every_seed': bool(deltas_zero.min() > 0.0),
        # A "real" margin, stated so a small positive number cannot be read as
        # a meaningful gait improvement.
        'beats_zero_by_margin_0.01': bool(mean_delta_zero > 0.01),
        'beats_zero_by_margin_0.05': bool(mean_delta_zero > 0.05),
        'beats_neutral_baseline': bool(deltas_neutral.mean() > 0.0),
    }


# --- Reporting helpers -----------------------------------------------------
def parse_seeds(text: str) -> tuple:
    """Parse a comma-separated seed list, rejecting anything non-integral."""
    seeds = []
    for chunk in str(text).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as exc:
            raise ValueError(f'Invalid seed {chunk!r}') from exc
        if value < 0:
            raise ValueError(f'Seeds must be non-negative, got {value}')
        seeds.append(value)
    if not seeds:
        raise ValueError('At least one seed is required')
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Duplicate seeds in {text!r}')
    return tuple(seeds)
