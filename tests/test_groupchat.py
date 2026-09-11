"""Decision-point construction: labels, causality, and determinism."""
import numpy as np
import pytest

from notflyguy.connectome import DRIVE_CHANNELS
from notflyguy.groupchat import (CONTEXT_NAMES, ChatStateTracker, build_decision_points,
                                 dataset_stats, drive_for_agent, drive_for_user,
                                 feature_matrix, feature_names)

requires_data = pytest.mark.skipif(
    not __import__('pathlib').Path('data/malecns-v1.0/motor-circuit.npz').exists(),
    reason='measured circuit not extracted',
)


def test_drive_vectors_match_the_circuit_contract():
    assert drive_for_user('hello', False, True, 1.0, 0).shape == (DRIVE_CHANNELS,)
    assert drive_for_agent('hi there').shape == (DRIVE_CHANNELS,)
    assert np.isfinite(drive_for_user('x' * 5000, True, False, 1e9, 99)).all()


@requires_data
def test_labels_follow_the_transcript(decision_points):
    # group-1: reply, reply, no reply (user-user), reply; group-2: reply.
    assert len(decision_points) == 5
    responded = [p.respond for p in decision_points]
    assert responded == [True, True, False, True, True]
    assert dataset_stats(decision_points)['responded'] == 4


@requires_data
def test_silent_decisions_carry_no_length_label(decision_points):
    silent = [p for p in decision_points if not p.respond]
    assert silent
    assert all(p.reply_length == -1.0 for p in silent)
    _, _, y_length = feature_matrix(decision_points)
    assert np.isnan(y_length).sum() == len(silent)


@requires_data
def test_features_are_causal(decision_points, group_events):
    """A prefix of the transcript must produce identical earlier features.

    If any feature peeked at a later message this would fail.
    """
    first_session = group_events[0].session_id
    rows = [e for e in group_events if e.session_id == first_session]
    full = build_decision_points(rows, seed=0)
    for cut in range(1, len(rows) + 1):
        partial = build_decision_points(rows[:cut], seed=0)
        for full_point, partial_point in zip(full[:len(partial)], partial):
            assert full_point.context == pytest.approx(partial_point.context)
            assert full_point.connectome == pytest.approx(partial_point.connectome)


@requires_data
def test_decision_points_are_deterministic(group_events):
    first = build_decision_points(group_events, seed=0)
    second = build_decision_points(group_events, seed=0)
    assert [p.context for p in first] == [p.context for p in second]
    assert [p.connectome for p in first] == [p.connectome for p in second]


@requires_data
def test_live_tracker_matches_offline_features(group_events):
    """Training and serving must produce byte-identical features.

    The dashboard replays events through a bare ChatStateTracker while the
    trainer uses build_decision_points. If those two ever diverge, a trained
    policy would be served inputs it never saw.
    """
    from notflyguy.connectome import ConnectomeState
    session_id = group_events[0].session_id
    rows = [e for e in group_events if e.session_id == session_id]
    offline = build_decision_points(rows, seed=0)
    tracker = ChatStateTracker(ConnectomeState(seed=0), session_id=session_id)
    replayed = []
    for event in rows:
        if event.is_agent:
            tracker.observe_agent(event.text, event.timestamp)
        else:
            replayed.append(tracker.observe_user(event))
    assert len(replayed) == len(offline)
    for (context, connectome), point in zip(replayed, offline):
        assert context == pytest.approx(point.context)
        assert connectome == pytest.approx(point.connectome)


@requires_data
def test_tracker_reset_clears_state(group_events):
    from notflyguy.connectome import ConnectomeState
    tracker = ChatStateTracker(ConnectomeState(seed=0))
    tracker.observe_user(group_events[0])
    assert tracker.seen_senders
    tracker.reset()
    assert not tracker.seen_senders
    assert np.allclose(tracker.brain.activity, 0.0)


def test_feature_names_match_matrix_width(decision_points):
    X, _, _ = feature_matrix(decision_points, include_connectome=True)
    assert X.shape[1] == len(feature_names(include_connectome=True))
    X_context, _, _ = feature_matrix(decision_points, include_connectome=False)
    assert X_context.shape[1] == len(feature_names(include_connectome=False))
    assert len(CONTEXT_NAMES) == X_context.shape[1]


@requires_data
def test_dataset_stats_report_group_coverage(decision_points):
    stats = dataset_stats(decision_points)
    assert stats['group_decision_points'] == len(decision_points)
    assert stats['decision_points'] == 5
    assert 0.0 <= stats['respond_rate'] <= 1.0
