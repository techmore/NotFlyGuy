"""Motor control over the measured circuit; not biological emulation.

The controller is a thin adapter: :class:`~notflyguy.connectome.ConnectomeState`
owns the measured recurrent structure, and
:class:`~notflyguy.motor_projection.LegMotorReadout` owns the modeled mapping
from motor neurons to FlyBody actuators.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .connectome import ConnectomeState
from .motor_projection import LegMotorReadout

# The walk_on_ball task's reward target. Kept here so the controller and the
# trainer cannot disagree about what the fly is being asked to do.
TARGET_BALL_VELOCITY = np.array([0.0, -5.0, 0.0])
VELOCITY_SCALE = 5.0
DRIVE_CLIP = 2.0


class MotorCircuitController:
    """Drive the circuit from ball velocity and read actuators off motor neurons."""

    def __init__(self, root, actuator_names, seed=0, decay=0.2, drive_scale=6.0,
                 action_limits=None):
        root = Path(root)
        catalog = root / 'leg-motor-catalog.json'
        if not catalog.exists():
            raise FileNotFoundError(
                f'Missing {catalog}. Run scripts/catalog_leg_neurons.py first.'
            )
        self.brain = ConnectomeState(root, seed=seed, gain=0.8, decay=decay,
                                     drive_scale=drive_scale, drive_channels=3)
        neurons = json.loads(catalog.read_text())['neurons']
        self.readout = LegMotorReadout(neurons, actuator_names, seed, limits=action_limits)

    # Circuit views kept for the dashboard and for existing callers.
    @property
    def ids(self):
        return self.brain.body_ids

    @property
    def motor_indices(self):
        return self.brain.motor_indices

    @property
    def upstream_indices(self):
        return self.brain.upstream_indices

    @property
    def recurrent(self):
        return self.brain.recurrent

    @property
    def activity(self):
        return self.brain.activity

    def reset(self) -> None:
        self.brain.reset()

    def velocity_error(self, ball_velocity) -> np.ndarray:
        velocity = np.asarray(ball_velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('Expected three finite ball velocity components')
        return np.clip((TARGET_BALL_VELOCITY - velocity) / VELOCITY_SCALE, -DRIVE_CLIP, DRIVE_CLIP)

    def step(self, ball_velocity) -> np.ndarray:
        """Modeled sensory injection, not identified biological sensory cells."""
        drive = self.velocity_error(ball_velocity)
        self.brain.step(drive)
        return self.readout(self.brain.motor_activity)
