"""Modeled leg-local readout, not a verified muscle-to-actuator mapping."""
import numpy as np


class LegMotorReadout:
    """Trainable weights only between neurons and actuators of the same leg.

    Biology supplies the neuron segment/side. Individual weights are modeling
    parameters: the connectome does not map motor neurons to MuJoCo controls.
    Head, abdomen, and adhesion remain at zero in this initial experiment.
    """

    def __init__(self, neurons, actuator_names, seed=0):
        self.mask = np.zeros((len(actuator_names), len(neurons)), dtype=bool)
        for i, name in enumerate(actuator_names):
            if 'adhere' in name:
                continue
            for j, neuron in enumerate(neurons):
                segment = neuron['somaNeuromere']
                side = {'L': 'left', 'R': 'right'}.get(neuron['somaSide'])
                if segment in ('T1', 'T2', 'T3') and side and name.endswith(f'_{segment}_{side}'):
                    self.mask[i, j] = True
        self.weights = np.random.default_rng(seed).normal(0, .01, self.mask.shape) * self.mask

    def __call__(self, motor_activity):
        activity = np.asarray(motor_activity, dtype=float)
        if activity.shape != (self.mask.shape[1],) or not np.isfinite(activity).all():
            raise ValueError('Expected a finite activity value for each motor neuron')
        return np.tanh((self.weights * self.mask) @ activity)
