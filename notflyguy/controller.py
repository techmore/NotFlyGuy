"""Replaceable controller boundary between a brain and a simulated body."""

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True)
class FlyObservation:
    """Signals available to a controller at one simulation step."""

    timestep: int
    sensors: tuple[float, ...] = ()
    reward: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FlyAction:
    """Motor command consumed by a body adapter."""

    motors: tuple[float, ...]


class Controller(Protocol):
    def step(self, observation: FlyObservation) -> FlyAction:
        """Convert one body observation into a motor command."""


class DeterministicController:
    """Small placeholder used to validate the integration contract."""

    def __init__(self, motor_count: int = 59) -> None:
        self.motor_count = motor_count

    def step(self, observation: FlyObservation) -> FlyAction:
        del observation
        return FlyAction(motors=tuple(0.0 for _ in range(self.motor_count)))


def action_from_sequence(values: Sequence[float]) -> FlyAction:
    return FlyAction(motors=tuple(float(value) for value in values))

