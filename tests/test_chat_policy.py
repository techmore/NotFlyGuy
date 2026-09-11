"""Policy training, metrics, persistence, and the honesty gate."""
import numpy as np
import pytest

from notflyguy.chat_policy import (ChatPolicy, accuracy, brier_score, majority_baseline,
                                   mean_absolute_error, roc_auc, sigmoid)


def test_sigmoid_is_stable_at_extremes():
    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert values[0] < 1e-15
    assert values[1] == pytest.approx(0.5)
    assert values[2] > 1 - 1e-15
    assert np.isfinite(values).all()


def test_accuracy_and_majority_baseline():
    y = np.array([1, 1, 1, 0])
    assert accuracy(y, np.array([0.9, 0.9, 0.9, 0.9])) == pytest.approx(0.75)
    assert majority_baseline(y) == pytest.approx(0.75)
    assert majority_baseline(np.array([1, 0])) == pytest.approx(0.5)


def test_roc_auc_handles_perfect_separation_and_ties():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_roc_auc_is_undefined_for_a_single_class():
    assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_brier_and_mae():
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert mean_absolute_error(np.array([1.0, np.nan]), np.array([0.0, 0.5])) == pytest.approx(1.0)


def test_policy_learns_a_separable_pattern():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    policy = ChatPolicy(6, l2=0.01).fit(X, y)
    assert accuracy(y, policy.predict_proba(X)) > 0.9
    assert roc_auc(y, policy.predict_proba(X)) > 0.95


def test_policy_is_invariant_to_feature_scaling():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = (X[:, 2] > 0).astype(float)
    scaled = X * np.array([1000.0, 0.001, 1.0, 50.0])
    plain = ChatPolicy(4, l2=0.01).fit(X, y)
    rescaled = ChatPolicy(4, l2=0.01).fit(scaled, y)
    assert accuracy(y, plain.predict_proba(X)) == pytest.approx(
        accuracy(y, rescaled.predict_proba(scaled)))


def test_length_head_trains_and_ignores_nan_targets():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(float)
    lengths = X[:, 1] * 0.2 + 0.5
    lengths[::3] = np.nan
    policy = ChatPolicy(5, l2=0.01, length_l2=0.01).fit(X, y, lengths)
    prediction = policy.predict_length(X)
    assert prediction.shape == (200,)
    assert np.isfinite(prediction).all()


def test_fit_rejects_mismatched_input():
    policy = ChatPolicy(3)
    with pytest.raises(ValueError):
        policy.fit(np.zeros((4, 3)), np.zeros(5))
    with pytest.raises(ValueError):
        policy.fit(np.zeros((0, 3)), np.zeros(0))


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] > 0).astype(float)
    policy = ChatPolicy(5, l2=0.1, names=('a', 'b', 'c', 'd', 'e')).fit(X, y, X[:, 1])
    path = policy.save(tmp_path / 'nested' / 'policy.npz')
    restored = ChatPolicy.load(path)
    assert restored.n_features == 5
    assert restored.names == ('a', 'b', 'c', 'd', 'e')
    assert np.allclose(policy.predict_proba(X), restored.predict_proba(X))
    assert np.allclose(policy.predict_length(X), restored.predict_length(X))


def test_load_missing_policy_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChatPolicy.load(tmp_path / 'absent.npz')


def test_describe_reports_influential_features():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = (X[:, 1] > 0).astype(float)
    policy = ChatPolicy(3, l2=0.01, names=('weak', 'strong', 'noise')).fit(X, y)
    top = dict(policy.describe()['top_respond_features'])
    assert abs(top['strong']) > abs(top['weak'])
