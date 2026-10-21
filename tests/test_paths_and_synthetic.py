"""Data-root resolution and the synthetic circuit's format contract."""
import json

import numpy as np
import pytest

from notflyguy import motor_training as mt
from notflyguy.connectome import ConnectomeState, load_summary
from notflyguy.paths import CATALOG_FILE, CIRCUIT_FILE, DEFAULT_DATA_ROOT, ENV_DATA_ROOT, SUMMARY_FILE, data_root
from notflyguy.synthetic import synthetic_catalog, synthetic_graph, write_synthetic_circuit


# --- paths -------------------------------------------------------------------
def test_data_root_prefers_explicit_then_env_then_default(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_DATA_ROOT, raising=False)
    assert data_root() == DEFAULT_DATA_ROOT
    monkeypatch.setenv(ENV_DATA_ROOT, str(tmp_path / 'env'))
    assert data_root() == tmp_path / 'env'
    assert data_root(tmp_path / 'explicit') == tmp_path / 'explicit'
    monkeypatch.setenv(ENV_DATA_ROOT, '   ')
    assert data_root() == DEFAULT_DATA_ROOT


def test_env_override_is_read_at_call_time(tmp_path, monkeypatch):
    """Setting the variable after import must still take effect (tests rely on it)."""
    write_synthetic_circuit(tmp_path, n_motor=12, n_upstream=8)
    monkeypatch.setenv(ENV_DATA_ROOT, str(tmp_path))
    brain = ConnectomeState()
    assert len(brain) == 20
    assert load_summary()['synthetic'] is True


def test_missing_circuit_error_mentions_the_override(tmp_path):
    with pytest.raises(Exception) as info:
        ConnectomeState(tmp_path / 'nowhere')
    assert ENV_DATA_ROOT in str(info.value)


# --- synthetic circuit -------------------------------------------------------
def test_synthetic_files_match_the_measured_layout(tmp_path):
    summary = write_synthetic_circuit(tmp_path, n_motor=30, n_upstream=20, seed=3)
    for name in (CIRCUIT_FILE, CATALOG_FILE, SUMMARY_FILE):
        assert (tmp_path / name).exists(), name
    with np.load(tmp_path / CIRCUIT_FILE, allow_pickle=False) as graph:
        assert set(graph.files) == {'body_ids', 'pre', 'post', 'synapse_counts',
                                    'motor_indices', 'upstream_indices'}
        assert len(graph['motor_indices']) == 30
        assert len(graph['upstream_indices']) == 20
        assert len(graph['pre']) == len(graph['post']) == len(graph['synapse_counts'])
        assert summary['edges'] == len(graph['pre'])
    catalog = json.loads((tmp_path / CATALOG_FILE).read_text())
    assert catalog['count'] == len(catalog['neurons']) == 30
    assert summary['synthetic'] is True
    assert json.loads((tmp_path / SUMMARY_FILE).read_text())['synthetic'] is True


def test_synthetic_graph_has_no_self_loops_or_duplicates_and_covers_motors():
    graph = synthetic_graph(n_motor=40, n_upstream=16, edges_per_node=5, seed=1)
    pre, post = graph['pre'], graph['post']
    assert not np.any(pre == post)
    assert len(set(zip(pre.tolist(), post.tolist()))) == len(pre)
    assert np.all(graph['synapse_counts'] >= 1)
    incoming = set(post.tolist())
    assert all(int(m) in incoming for m in graph['motor_indices'])


def test_synthetic_graph_is_deterministic_per_seed():
    a = synthetic_graph(n_motor=10, n_upstream=6, seed=7)
    b = synthetic_graph(n_motor=10, n_upstream=6, seed=7)
    c = synthetic_graph(n_motor=10, n_upstream=6, seed=8)
    assert np.array_equal(a['post'], b['post'])
    assert np.array_equal(a['synapse_counts'], b['synapse_counts'])
    assert not np.array_equal(a['synapse_counts'], c['synapse_counts'])


def test_synthetic_catalog_assigns_every_neuron_to_a_leg():
    catalog = synthetic_catalog(n_motor=13)
    for neuron in catalog['neurons']:
        assert neuron['somaNeuromere'] in ('T1', 'T2', 'T3')
        assert neuron['somaSide'] in ('L', 'R')
    assert sum(catalog['counts_by_subclass'].values()) == 13


def test_default_synthetic_catalog_reproduces_the_frozen_readout_size(tmp_path):
    """381 leg neurons x 8 trainable actuators per leg = 3048 weights, as measured."""
    write_synthetic_circuit(tmp_path)
    readout = mt.make_readout(root=tmp_path, seed=0)
    assert int(readout.mask.sum()) == mt.N_WEIGHT_PARAMETERS
    assert readout.parameters().shape == (mt.N_PARAMETERS,)


def test_synthetic_circuit_obeys_the_stability_bound(tmp_path):
    write_synthetic_circuit(tmp_path, n_motor=50, n_upstream=30)
    brain = ConnectomeState(tmp_path)
    row_sums = np.asarray(abs(brain.recurrent).sum(axis=1)).ravel()
    assert row_sums.max() <= brain.gain + 1e-9
    for _ in range(100):
        brain.step(np.ones(brain.drive_channels))
    assert np.isfinite(brain.activity).all()
    assert np.abs(brain.activity).max() <= 1.0 + 1e-9


def test_synthetic_rejects_empty_graphs():
    with pytest.raises(ValueError):
        synthetic_graph(n_motor=0)
