"""Trainable group-chat readout: connectome state + context -> behaviour.

Deliberately small and linear. The corpus is small, so a linear model with
explicit regularisation and standardised inputs is the honest choice: it can be
inspected, it cannot memorise the training sessions, and its held-out score is
meaningful. Two heads are trained:

* ``respond``  -- logistic probability that the agent speaks next.
* ``length``   -- ridge estimate of reply length, used to modulate verbosity.

Nothing here generates language. The policy only decides *whether* and *how
much* to speak; wording stays with the conversational model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        return float('nan')
    return float(np.mean((np.asarray(y_pred, dtype=float) > 0.5) == (y_true > 0.5)))


def majority_baseline(y_true) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        return float('nan')
    return float(max(np.mean(y_true > 0.5), 1.0 - np.mean(y_true > 0.5)))


def brier_score(y_true, probability) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        return float('nan')
    return float(np.mean((np.asarray(probability, dtype=float) - y_true) ** 2))


def roc_auc(y_true, probability) -> float:
    """Rank-based AUC with average ranks for ties."""
    y_true = np.asarray(y_true, dtype=float) > 0.5
    scores = np.asarray(probability, dtype=float)
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ordered = scores[order]
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[y_true].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def mean_absolute_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


class ChatPolicy:
    """Standardised linear policy with a respond head and a length head."""

    def __init__(self, n_features: int, l2: float = 1.0, length_l2: float = 10.0,
                 epochs: int = 3000, lr: float = 0.5, names=None):
        self.n_features = int(n_features)
        self.l2 = float(l2)
        self.length_l2 = float(length_l2)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.names = tuple(names) if names is not None else ()
        self.mean = np.zeros(self.n_features)
        self.scale = np.ones(self.n_features)
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0
        self.length_weights = np.zeros(self.n_features)
        self.length_bias = 0.0

    # -- internals ---------------------------------------------------------
    def _standardise(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f'Expected a matrix with {self.n_features} columns')
        return (X - self.mean) / self.scale

    def _fit_scaler(self, X) -> None:
        self.mean = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale = np.where(scale > EPS, scale, 1.0)

    # -- training ----------------------------------------------------------
    def fit(self, X, y_respond, y_length=None, sample_weight=None) -> ChatPolicy:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_respond, dtype=float)
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of rows')
        if X.shape[0] == 0:
            raise ValueError('Cannot fit on an empty dataset')
        self.n_features = X.shape[1]
        self._fit_scaler(X)
        Z = self._standardise(X)
        w = np.zeros(self.n_features)
        b = 0.0
        weights = (np.ones(len(y)) if sample_weight is None
                   else np.asarray(sample_weight, dtype=float))
        total = max(weights.sum(), EPS)
        for _ in range(self.epochs):
            p = sigmoid(Z @ w + b)
            residual = (p - y) * weights
            grad_w = Z.T @ residual / total + self.l2 * w
            grad_b = float(residual.sum() / total)
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.weights, self.bias = w, b

        if y_length is not None:
            target = np.asarray(y_length, dtype=float)
            mask = np.isfinite(target)
            if mask.sum() >= 2:
                Zl = Z[mask]
                tl = target[mask]
                gram = Zl.T @ Zl + self.length_l2 * np.eye(self.n_features)
                self.length_weights = np.linalg.solve(gram, Zl.T @ tl)
                self.length_bias = float(tl.mean() - Zl.mean(axis=0) @ self.length_weights)
        return self

    # -- inference ---------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        return sigmoid(self._standardise(X) @ self.weights + self.bias)

    def predict_length(self, X) -> np.ndarray:
        return self._standardise(X) @ self.length_weights + self.length_bias

    def describe(self) -> dict:
        """Most influential features, by standardised weight magnitude."""
        if self.names and len(self.names) == len(self.weights):
            pairs = sorted(zip(self.names, self.weights), key=lambda kv: -abs(kv[1]))
            return {'top_respond_features': [[n, round(float(w), 4)] for n, w in pairs[:10]],
                    'bias': round(float(self.bias), 4)}
        return {'bias': round(float(self.bias), 4)}

    # -- persistence -------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights, bias=np.array([self.bias]),
            length_weights=self.length_weights, length_bias=np.array([self.length_bias]),
            mean=self.mean, scale=self.scale,
            meta=np.array([json.dumps({
                'n_features': self.n_features,
                'names': list(self.names),
                'l2': self.l2, 'length_l2': self.length_l2,
                'format': 'notflyguy.chat_policy/1',
            })]),
        )
        return path

    @classmethod
    def load(cls, path) -> ChatPolicy:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'No trained group-chat policy at {path}')
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta'][0]))
            policy = cls(int(meta['n_features']), l2=float(meta.get('l2', 1.0)),
                         length_l2=float(meta.get('length_l2', 10.0)),
                         names=meta.get('names') or None)
            policy.weights = data['weights'].astype(float)
            policy.bias = float(data['bias'][0])
            policy.length_weights = data['length_weights'].astype(float)
            policy.length_bias = float(data['length_bias'][0])
            policy.mean = data['mean'].astype(float)
            policy.scale = data['scale'].astype(float)
        return policy
