"""Reusable recurrent state over a measured MaleCNS subgraph.

The circuit provides a fixed, data-derived dynamical system. It is *not* a
brain emulation and its activity is not a claim about the real fly. What the
connectome supplies is structure: which neurons exist, which are motor
neurons, and a measured coupling between them. Everything that maps external
signal into that structure, and everything read out of it, is an explicit
modeling choice recorded in the artifacts.

Stability note: the recurrent operator is row-normalized to total mass
``gain``. Its induced infinity norm is therefore ``gain``, so the spectral
radius is bounded by ``gain < 1`` and the linear part cannot diverge.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix, diags

from .paths import CIRCUIT_FILE, SUMMARY_FILE, data_root
from .paths import DEFAULT_DATA_ROOT as DEFAULT_ROOT  # noqa: F401  (kept for callers)

# Number of modeled sensory drive channels injected into upstream neurons.
DRIVE_CHANNELS = 5
# Names of those channels, in order. Documented so callers cannot silently
# reorder them without updating this contract. These are modeling choices, not
# identified biological sensory cell types.
DRIVE_NAMES = ('social_salience', 'novelty', 'recency_urgency', 'agent_pressure',
               'agent_reply')


class ConnectomeDataMissing(RuntimeError):
    """Raised when the measured circuit has not been fetched or extracted."""


class ConnectomeState:
    """Fixed recurrent dynamics with an explicit, replaceable input projection."""

    def __init__(self, root=None, seed=0, gain=0.8, decay=0.8,
                 drive_scale=6.0, drive_channels=DRIVE_CHANNELS,
                 circuit_file=CIRCUIT_FILE):
        root = data_root(root)
        circuit = root / circuit_file
        if not circuit.exists():
            raise ConnectomeDataMissing(
                f'Missing {circuit}. Run scripts/fetch_connectome.py, then '
                'scripts/catalog_leg_neurons.py, then scripts/extract_motor_circuit.py '
                '(or python scripts/bootstrap_data.py for all three), or point '
                'NOTFLYGUY_DATA_ROOT at a directory that already holds the circuit.'
            )
        graph = np.load(circuit, allow_pickle=False)
        self.body_ids = graph['body_ids']
        self.motor_indices = graph['motor_indices']
        self.upstream_indices = graph['upstream_indices']
        counts = graph['synapse_counts'].astype(float)
        matrix = csr_matrix(
            (counts, (graph['post'], graph['pre'])),
            shape=(len(self.body_ids), len(self.body_ids)),
        )
        # Nonnegative anatomical coupling, row-normalized to bounded mass.
        # No neurotransmitter-derived physiological signs are claimed here.
        self.recurrent = diags(gain / np.maximum(np.asarray(matrix.sum(axis=1)).ravel(), 1)) @ matrix
        self.activity = np.zeros(len(self.body_ids), dtype=float)
        self.gain = float(gain)
        self.decay = float(decay)
        self.drive_channels = int(drive_channels)
        # Fixed projection onto upstream neurons. Column-normalized so every
        # channel carries comparable influence before the scale is applied.
        projection = np.zeros((len(self.body_ids), self.drive_channels), dtype=float)
        upstream = np.random.default_rng(seed).normal(0, 1, (len(self.upstream_indices), self.drive_channels))
        norms = np.linalg.norm(upstream, axis=0)
        upstream = upstream / np.where(norms > 0, norms, 1.0)
        projection[self.upstream_indices] = upstream * float(drive_scale)
        self.input_projection = projection

    def __len__(self) -> int:
        return len(self.body_ids)

    @property
    def edge_count(self) -> int:
        return int(self.recurrent.nnz)

    def reset(self) -> None:
        self.activity.fill(0.0)

    def step(self, drive) -> np.ndarray:
        """Advance one step under a finite drive vector; return the activity."""
        signal = np.asarray(drive, dtype=float)
        if signal.shape != (self.drive_channels,):
            raise ValueError(f'Expected {self.drive_channels} drive components')
        if not np.isfinite(signal).all():
            raise ValueError('Drive components must be finite')
        target = np.tanh(self.recurrent @ self.activity + self.input_projection @ signal)
        self.activity = (1.0 - self.decay) * self.activity + self.decay * target
        return self.activity

    @property
    def motor_activity(self) -> np.ndarray:
        return self.activity[self.motor_indices]

    def features(self) -> np.ndarray:
        """Compact, low-dimensional descriptor of the current neural state.

        Deliberately small: it is the only connectome signal the chat policy
        sees, which keeps the trainable parameter count honest on small data.
        """
        activity = self.activity
        motor = activity[self.motor_indices]
        upstream = activity[self.upstream_indices]
        return np.array([
            float(np.mean(activity)),
            float(np.mean(np.abs(activity))),
            float(np.sqrt(np.mean(activity ** 2))),
            float(np.mean(motor)),
            float(np.mean(np.abs(motor))),
            float(np.mean(upstream)),
            float(np.mean(np.abs(upstream))),
            float(np.max(activity)),
        ], dtype=float)

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return ('mean_activity', 'mean_abs_activity', 'activity_rms', 'mean_motor',
                'mean_abs_motor', 'mean_upstream', 'mean_abs_upstream', 'peak_activity')


def load_summary(root=None) -> dict:
    """Read the recorded extraction summary, if present."""
    path = data_root(root) / SUMMARY_FILE
    if not path.exists():
        raise ConnectomeDataMissing(f'Missing {path}. Run scripts/extract_motor_circuit.py.')
    return json.loads(path.read_text())
