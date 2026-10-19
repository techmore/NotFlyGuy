"""Small, inspectable online policy readout for the DragonRealms adapter.

The connectome remains fixed.  This readout is the trainable part: it learns
which already-validated proposal is useful in a given connectome/game state.
It is deliberately linear and persisted as JSON so training is auditable.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


class LinearActionLearner:
    def __init__(self, path: str | Path | None = None, *, alpha=.08,
                 epsilon=.12, seed=0):
        self.path = Path(path) if path else None
        self.alpha, self.epsilon = float(alpha), float(epsilon)
        self.rng = random.Random(seed)
        self.weights: dict[str, list[float]] = {}
        self.updates = 0
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self.weights = {str(k): [float(x) for x in v]
                            for k, v in data.get('weights', {}).items()}
            self.updates = int(data.get('updates', 0))

    @staticmethod
    def _x(features):
        values = [1.0] + [float(x) for x in features]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('features must be finite')
        return values

    def _row(self, action, size):
        row = self.weights.setdefault(action, [0.0] * size)
        if len(row) != size:
            raise ValueError('feature size changed for persisted learner')
        return row

    def score(self, action, features):
        x = self._x(features)
        return sum(a * b for a, b in zip(self._row(action, len(x)), x))

    def choose(self, actions, features, *, explore=True):
        actions = tuple(dict.fromkeys(actions))
        if not actions:
            raise ValueError('at least one action is required')
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(actions), True
        # Stable first-action tie break keeps a fresh checkpoint on the
        # planner's useful proposal instead of lexicographically preferring
        # ``wait`` before it has observed any reward.
        scores = [(self.score(a, features), a) for a in actions]
        return max(scores, key=lambda item: item[0])[1], False

    def update(self, action, features, reward):
        x = self._x(features)
        row = self._row(action, len(x))
        prediction = sum(a * b for a, b in zip(row, x))
        error = float(reward) - prediction
        for i, value in enumerate(x):
            row[i] += self.alpha * error * value / (1.0 + sum(v * v for v in x))
        self.updates += 1
        self.save()
        return {'prediction': prediction, 'error': error, 'updates': self.updates}

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({'schema': 'notflyguy.dr.learner/1',
                                         'updates': self.updates,
                                         'weights': self.weights}, indent=2))
