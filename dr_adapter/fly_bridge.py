"""Small public bridge from DR observations into the downloaded Fly circuit."""
from __future__ import annotations

import numpy as np

from .observation import Observation


class FlyBridge:
    namespace = 'notflyguy.dr_adapter'
    def __init__(self, circuit_root=None, seed=0, checkpoint=None, learning_rate=.002):
        from notflyguy.connectome import ConnectomeState
        # ``None`` resolves to NOTFLYGUY_DATA_ROOT or the in-repo default.
        self.brain = ConnectomeState(circuit_root or None, seed=seed, drive_channels=5)
        self.checkpoint = checkpoint
        self.learning_rate = float(learning_rate)
        self.updates = 0
        self.last_drive = np.zeros(5, dtype=float)
        if checkpoint:
            try:
                data = np.load(checkpoint, allow_pickle=False)
                if data['input_projection'].shape == self.brain.input_projection.shape:
                    self.brain.input_projection = data['input_projection']
                    self.updates = int(data.get('updates', 0))
            except (OSError, KeyError, ValueError):
                pass
    def observe(self, observation: Observation) -> dict:
        drive = np.array([
            1.0 if observation.in_combat is True else 0.0,
            1.0 if observation.target_present is True else 0.0,
            float(np.clip(1.0 - observation.health_fraction, 0, 1)) if observation.health_fraction is not None else 1.0,
            float(np.clip(float(observation.roundtime_seconds or 0) / 10.0, 0, 1)),
            # Modeled tonic input keeps ordinary field decisions observable
            # to the recurrent substrate. This is not a biological claim.
            1.0 if observation.status != 'ready' else .1,
        ], dtype=float)
        self.last_drive = drive
        self.brain.step(drive)
        return {'namespace': self.namespace, 'drive_mapping': 'dr-sensory/2', 'drive': drive.tolist(), 'features': [round(float(x), 6) for x in self.brain.features()],
                'activity_rms': float(np.sqrt(np.mean(self.brain.activity ** 2)))}

    def learn(self, reward: float) -> dict:
        """Reward-modulated plasticity on the DR sensory projection.

        The extracted recurrent graph is unchanged; this adapter-specific
        projection is the learned DR connectome interface and is checkpointed
        separately from the downloaded anatomy.
        """
        signal = np.tanh(self.brain.activity)
        delta = self.learning_rate * float(reward) * np.outer(signal, self.last_drive)
        mask = np.zeros_like(delta); mask[self.brain.upstream_indices] = 1.0
        previous = self.brain.input_projection
        self.brain.input_projection = np.clip(self.brain.input_projection + delta * mask, -8.0, 8.0)
        self.updates += 1
        if self.checkpoint:
            np.savez_compressed(self.checkpoint, input_projection=self.brain.input_projection,
                                updates=np.array(self.updates))
        return {'updates': self.updates, 'weight_delta_rms': float(np.sqrt(np.mean((self.brain.input_projection - previous) ** 2)))}
