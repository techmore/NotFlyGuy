"""Experimental rate dynamics over measured edges; not biological emulation."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags
from .motor_projection import LegMotorReadout


class MotorCircuitController:
    def __init__(self, root, actuator_names, seed=0):
        root = Path(root)
        graph = np.load(root / 'motor-circuit.npz', allow_pickle=False)
        self.ids = graph['body_ids']
        self.motor_indices = graph['motor_indices']
        self.upstream_indices = graph['upstream_indices']
        counts = graph['synapse_counts'].astype(float)
        matrix = csr_matrix((counts, (graph['post'], graph['pre'])), shape=(len(self.ids),len(self.ids)))
        # Nonnegative anatomical coupling, row-normalized for bounded dynamics.
        # No neurotransmitter-derived physiological signs are claimed here.
        self.recurrent = diags(.8 / np.maximum(np.asarray(matrix.sum(axis=1)).ravel(), 1)) @ matrix
        self.activity = np.zeros(len(self.ids))
        self.input_weights = np.zeros((len(self.ids), 3))
        self.input_weights[self.upstream_indices] = np.random.default_rng(seed).normal(0, .1, (len(self.upstream_indices),3))
        neurons = json.loads((root / 'leg-motor-catalog.json').read_text())['neurons']
        self.readout = LegMotorReadout(neurons, actuator_names, seed)

    def reset(self):
        self.activity.fill(0)

    def step(self, ball_velocity):
        velocity = np.asarray(ball_velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('Expected three finite ball velocity components')
        # Modeled target-error injection, not identified biological sensory cells.
        error = np.clip((np.array([0., -5., 0.]) - velocity) / 5, -2, 2)
        target = np.tanh(self.recurrent @ self.activity + self.input_weights @ error)
        self.activity = .8 * self.activity + .2 * target
        return self.readout(self.activity[self.motor_indices])
