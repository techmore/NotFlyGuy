"""Modeled leg-local readout, not a verified muscle-to-actuator mapping.

Action ranges: ``walk_on_ball``'s ``action_spec()`` is **not** ``[0, 1]`` for
every actuator. Only the head and abdomen actuators span ``[0, 1]``; leg
actuators span their own ranges (for example ``coxa_T1_left`` is ``[-3, 3]``).
A fixed ``(0, 1)`` output map therefore makes most of the reachable posture
space unreachable. The readout is an affine map onto each actuator's declared
range, taken from the live spec by the caller.
"""
import numpy as np

# Defaults used only when a caller supplies no actuator limits.
ACTION_MIN = 0.0
ACTION_MAX = 1.0
ACTION_NEUTRAL = 0.5


def normalize_limits(limits, count):
    """Return a finite ``(count, 2)`` array of ``[min, max]`` per actuator."""
    if limits is None:
        return np.tile(np.array([[ACTION_MIN, ACTION_MAX]], dtype=float), (count, 1))
    array = np.asarray(limits, dtype=float)
    if array.ndim == 1:
        if array.shape != (2,):
            raise ValueError(f'Expected a (2,) range, got shape {array.shape}')
        array = np.tile(array, (count, 1))
    if array.shape != (count, 2):
        raise ValueError(f'Expected limits shaped ({count}, 2) or (2,), got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError('Actuator limits must be finite')
    if np.any(array[:, 1] <= array[:, 0]):
        raise ValueError('Each actuator needs max > min')
    return array


def limits_from_spec(spec):
    """Per-actuator ``[min, max]`` from a dm_control/MuJoCo action spec."""
    minimum = np.asarray(spec.minimum, dtype=float)
    maximum = np.asarray(spec.maximum, dtype=float)
    if minimum.shape != maximum.shape:
        raise ValueError('Action spec minimum and maximum shapes differ')
    return np.stack([minimum, maximum], axis=1)


def to_action(drive, limits=None):
    """Map an unbounded drive in (-inf, inf) onto each actuator's range.

    With no ``limits`` this falls back to a single ``(0, 1)`` range, which is
    not the live FlyBody spec and is only suitable for scalar checks.
    """
    drive = np.asarray(drive, dtype=float)
    if limits is None:
        return ACTION_NEUTRAL + (ACTION_MAX - ACTION_MIN) / 2.0 * np.tanh(drive)
    array = np.asarray(limits, dtype=float)
    center = array.mean(axis=1)
    half = (array[:, 1] - array[:, 0]) / 2.0
    return center + half * np.tanh(drive)


class LegMotorReadout:
    """Trainable weights only between neurons and actuators of the same leg.

    Biology supplies the neuron segment/side. Individual weights are modeling
    parameters: the connectome does not map motor neurons to MuJoCo controls.
    Head, abdomen, and adhesion remain untrained in this experiment.

    ``bias`` is a tonic drive per actuator. Without it the readout can only
    produce commands proportional to neural activity, which the earlier
    circuit kept near zero, leaving the policy with no control authority at
    all -- the reason training could not outperform doing nothing.
    """

    def __init__(self, neurons, actuator_names, seed=0, limits=None):
        self.actuator_names = tuple(actuator_names)
        self.limits = normalize_limits(limits, len(self.actuator_names))
        self.center = self.limits.mean(axis=1)
        self.half = (self.limits[:, 1] - self.limits[:, 0]) / 2.0
        self.mask = np.zeros((len(actuator_names), len(neurons)), dtype=bool)
        for i, name in enumerate(actuator_names):
            if 'adhere' in name:
                continue
            for j, neuron in enumerate(neurons):
                segment = neuron['somaNeuromere']
                side = {'L': 'left', 'R': 'right'}.get(neuron['somaSide'])
                if segment in ('T1', 'T2', 'T3') and side and name.endswith(f'_{segment}_{side}'):
                    self.mask[i, j] = True
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, .01, self.mask.shape) * self.mask
        self.bias = np.zeros(len(actuator_names), dtype=float)

    def parameters(self):
        """Flat, mask-respecting view used by the trainer."""
        return np.concatenate([self.weights[self.mask], self.bias])

    def set_parameters(self, flat, bias=True):
        flat = np.asarray(flat, dtype=float)
        if not np.isfinite(flat).all():
            raise ValueError('Parameters must be finite')
        count = int(self.mask.sum())
        expected = count + len(self.bias) if bias else count
        if flat.shape != (expected,):
            raise ValueError(f'Expected {expected} parameters, got {flat.shape}')
        self.weights = np.zeros(self.mask.shape, dtype=float)
        self.weights[self.mask] = flat[:count]
        if bias:
            self.bias = flat[count:].copy()

    def __call__(self, motor_activity):
        activity = np.asarray(motor_activity, dtype=float)
        if activity.shape != (self.mask.shape[1],) or not np.isfinite(activity).all():
            raise ValueError('Expected a finite activity value for each motor neuron')
        drive = (self.weights * self.mask) @ activity + self.bias
        return self.center + self.half * np.tanh(drive)
