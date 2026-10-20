"""Wiring control: the shuffle must preserve exactly what it claims to."""
import numpy as np
import pytest

from notflyguy.wiring_control import _degrees, build_shuffled_circuit, evaluate, shuffle_edges


def _toy_graph():
    # A small directed graph with no self-loops or duplicate edges.
    pre = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    post = np.array([1, 2, 0, 3, 4, 5, 1, 2, 0, 3, 4, 5])
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    return pre, post, weights


# --- shuffle_edges -----------------------------------------------------------
def test_shuffle_preserves_in_and_out_degree():
    pre, post, _ = _toy_graph()
    n = 6
    out_before, in_before = _degrees(pre, post, n)
    result = shuffle_edges(pre, post, seed=0, swaps_per_edge=20)
    out_after, in_after = _degrees(pre, result['post'], n)
    assert np.array_equal(out_before, out_after)
    assert np.array_equal(in_before, in_after)


def test_shuffle_never_moves_edge_sources():
    """Weights travel with the source, so sources must be untouched."""
    pre, post, _ = _toy_graph()
    result = shuffle_edges(pre, post, seed=1, swaps_per_edge=20)
    assert np.array_equal(pre, pre)
    assert result['post'].shape == post.shape


def test_shuffle_creates_no_self_loops_or_duplicates():
    pre, post, _ = _toy_graph()
    result = shuffle_edges(pre, post, seed=2, swaps_per_edge=50)
    new_post = result['post']
    assert not np.any(pre == new_post)
    assert len(set(zip(pre.tolist(), new_post.tolist()))) == len(pre)


def test_shuffle_is_deterministic_and_seed_dependent():
    pre, post, _ = _toy_graph()
    assert np.array_equal(shuffle_edges(pre, post, seed=3)['post'],
                          shuffle_edges(pre, post, seed=3)['post'])
    assert not np.array_equal(shuffle_edges(pre, post, seed=3)['post'],
                              shuffle_edges(pre, post, seed=4)['post'])


def test_shuffle_actually_changes_wiring():
    pre, post, _ = _toy_graph()
    result = shuffle_edges(pre, post, seed=5, swaps_per_edge=20)
    assert result['accepted'] > 0
    assert np.any(result['post'] != post)


def test_zero_swaps_is_identity():
    pre, post, _ = _toy_graph()
    result = shuffle_edges(pre, post, seed=0, swaps_per_edge=0)
    assert np.array_equal(result['post'], post)
    assert result['accepted'] == 0


# --- build_shuffled_circuit --------------------------------------------------
@pytest.fixture
def toy_circuit(tmp_path):
    path = tmp_path / 'motor-circuit.npz'
    pre, post, weights = _toy_graph()
    np.savez_compressed(
        path,
        body_ids=np.arange(6, dtype=np.int64),
        pre=pre, post=post, synapse_counts=weights,
        motor_indices=np.array([4, 5], dtype=np.int64),
        upstream_indices=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    return path


def test_build_shuffled_preserves_node_roles_and_keys(toy_circuit, tmp_path):
    dest = tmp_path / 'out' / 'shuffled.npz'
    meta = build_shuffled_circuit(toy_circuit, dest, seed=0)
    assert meta['degree_sequence_preserved'] is True
    with np.load(toy_circuit, allow_pickle=False) as original, \
            np.load(dest, allow_pickle=False) as shuffled:
        assert set(original.files) == set(shuffled.files)
        for key in ('body_ids', 'motor_indices', 'upstream_indices', 'pre', 'synapse_counts'):
            assert np.array_equal(original[key], shuffled[key]), key
        assert not np.array_equal(original['post'], shuffled['post'])


def test_build_shuffled_writes_a_loadable_circuit(toy_circuit, tmp_path):
    dest = tmp_path / 'shuffled.npz'
    meta = build_shuffled_circuit(toy_circuit, dest, seed=1, swaps_per_edge=20)
    assert meta['edges'] == 12
    assert 0.0 <= meta['edges_retargeted_fraction'] <= 1.0
    assert dest.exists()


# --- evaluate: orchestration only (no real dataset needed) -------------------
def test_evaluate_refuses_to_claim_on_a_tiny_corpus(circuit_root, group_events):
    """A 13-point corpus must never yield a wiring claim, however the AUCs order."""
    report = evaluate(circuit_root / 'motor-circuit.npz', group_events,
                      shuffle_seeds=(0,), folds=2)
    assert report['verdict']['sufficient_data'] is False
    assert report['verdict']['claim'].startswith('NOT ESTABLISHED')
    # Even if the ordering happened to favour the real wiring, the claim is withheld.
    assert 'must not be read as evidence' in report['verdict']['claim']


def test_evaluate_runs_and_compares_against_shuffles(circuit_root, group_events):
    report = evaluate(circuit_root / 'motor-circuit.npz', group_events,
                      shuffle_seeds=(0, 1), folds=2, workdir=None)
    assert report['real']['mean_roc_auc'] is not None
    assert len(report['shuffles']) == 2
    assert report['verdict']['shuffled_auc_min'] <= report['verdict']['shuffled_auc_max']
    assert isinstance(report['verdict']['real_beats_every_shuffle'], bool)
    assert report['verdict']['claim']
