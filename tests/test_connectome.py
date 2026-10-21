"""Circuit loading, stability bound, and feature extraction."""
import numpy as np
import pytest

from notflyguy.connectome import DRIVE_CHANNELS, DRIVE_NAMES, ConnectomeDataMissing, ConnectomeState, load_summary


def test_drive_contract_is_documented():
    assert DRIVE_CHANNELS == len(DRIVE_NAMES) == 5
    assert len(set(DRIVE_NAMES)) == DRIVE_CHANNELS


def test_recurrent_operator_cannot_diverge():
    brain = ConnectomeState()
    row_sums = np.asarray(abs(brain.recurrent).sum(axis=1)).ravel()
    assert row_sums.max() <= brain.gain + 1e-9
    # Induced infinity norm bounds the spectral radius, so a linear recurrence
    # cannot blow up. Check the empirical result too.
    brain.reset()
    for _ in range(200):
        brain.step(np.ones(DRIVE_CHANNELS))
    assert np.isfinite(brain.activity).all()
    assert np.abs(brain.activity).max() <= 1.0 + 1e-9


def test_zero_drive_leaves_the_circuit_quiet():
    brain = ConnectomeState()
    brain.reset()
    for _ in range(50):
        brain.step(np.zeros(DRIVE_CHANNELS))
    assert np.allclose(brain.activity, 0.0)


def test_drive_changes_activity():
    brain = ConnectomeState()
    brain.reset()
    for _ in range(30):
        brain.step(np.array([2.0, 1.0, 1.0, 2.0, 0.0]))
    driven = brain.features().copy()
    brain.reset()
    for _ in range(30):
        brain.step(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    other = brain.features()
    assert not np.allclose(driven, other)
    assert np.abs(driven).max() > 0.01


def test_step_validates_drive_shape_and_finiteness():
    brain = ConnectomeState()
    with pytest.raises(ValueError):
        brain.step(np.zeros(DRIVE_CHANNELS + 1))
    with pytest.raises(ValueError):
        brain.step(np.array([np.nan] * DRIVE_CHANNELS))


def test_feature_contract_is_consistent():
    brain = ConnectomeState()
    assert len(brain.features()) == len(brain.feature_names()) == 8
    assert brain.motor_activity.shape == brain.motor_indices.shape


def test_missing_data_raises_actionable_error(tmp_path):
    with pytest.raises(ConnectomeDataMissing) as info:
        ConnectomeState(tmp_path)
    assert 'fetch_connectome' in str(info.value)


def test_load_summary_reports_missing_file(tmp_path):
    with pytest.raises(ConnectomeDataMissing):
        load_summary(tmp_path)
