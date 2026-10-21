"""Corpus readiness: gates, projections, and text-free reporting."""
import json
from datetime import datetime, timezone

import pytest

from notflyguy.chatlog import ChatEvent, pseudonym
from notflyguy.connectome import ConnectomeState
from notflyguy.corpus import DEFAULT_GATES, project, readiness, status, summarise, weekly_counts
from notflyguy.groupchat import DecisionPoint


def _events(count, session_id, sender, start, spacing_hours=1.0, chat_type='group'):
    return [
        ChatEvent(
            session_id=session_id, group_id='g-' + session_id,
            sender_id=pseudonym(sender), sender_name='owner',
            text='hello there', timestamp=start + index * spacing_hours * 3600,
            is_agent=False, addressed_to_agent=False, index=index,
            chat_type=chat_type, source='signal',
        )
        for index in range(count)
    ]


def test_readiness_reports_every_unmet_gate():
    report = readiness({'sessions': 1, 'group_decision_points': 4}, {'participants': 1})
    assert report['ready'] is False
    assert set(report['blockers']) == {
        'group_sessions: 1/5', 'group_decision_points: 4/50', 'group_participants: 1/3'}


def test_readiness_is_met_only_when_all_gates_pass():
    stats = {'sessions': 5, 'group_decision_points': 50}
    corpus = {'participants': 3}
    assert readiness(stats, corpus)['ready'] is True
    assert readiness(dict(stats, sessions=4), corpus)['ready'] is False


def test_gate_overrides_are_respected():
    report = readiness({'sessions': 2, 'group_decision_points': 10},
                       {'participants': 1},
                       gates={'group_sessions': 2, 'group_decision_points': 5,
                              'group_participants': 1})
    assert report['ready'] is True
    assert report['gates']['group_sessions'] == 2


def test_default_gates_require_multiple_sessions():
    """A single long conversation must never be sufficient."""
    assert DEFAULT_GATES['group_sessions'] >= 5


# --- weekly_counts / project -------------------------------------------------
def _points(timestamps):
    from notflyguy.groupchat import CONTEXT_NAMES
    size_context, size_connectome = len(CONTEXT_NAMES), len(ConnectomeState.feature_names())
    return [
        DecisionPoint(session_id='s', group_id='g', sender_id='x', timestamp=ts, index=i,
                      context=(0.0,) * size_context, connectome=(0.0,) * size_connectome,
                      respond=True, reply_length=0.5, addressed=False, is_group=True)
        for i, ts in enumerate(timestamps)
    ]


def test_weekly_counts_bucket_by_iso_week():
    base = datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp()  # a Monday
    points = _points([base, base + 3600, base + 8 * 86400])
    weekly = weekly_counts(points)
    assert len(weekly) == 2
    assert sorted(weekly.values()) == [1, 2]


def test_project_extrapolates_from_recent_weeks():
    result = project({'w1': 2, 'w2': 4, 'w3': 6}, need=50, have=12)
    assert result['rate_per_week'] == pytest.approx(4.0)
    assert result['weeks_to_gate'] == pytest.approx(9.5)
    assert 'not a forecast' in result['basis']


def test_project_handles_no_activity_and_no_data():
    assert project({}, need=50, have=0)['weeks_to_gate'] is None
    flat = project({'w1': 0, 'w2': 0}, need=50, have=3)
    assert flat['rate_per_week'] == 0.0
    assert flat['weeks_to_gate'] is None


def test_project_returns_zero_when_gate_already_met():
    assert project({'w1': 5}, need=50, have=50)['weeks_to_gate'] == 0.0


# --- status / summarise ------------------------------------------------------
def test_status_uses_injected_corpus_and_skips_all_sources():
    events = _events(6, 's1', '+15550001', 1_700_000_000)
    report = status(events=events)
    assert report['all_sources']['decision_points'] is None
    assert report['all_sources']['group_share'] is None
    assert report['group']['dataset']['decision_points'] == 6
    assert report['readiness']['ready'] is False


def test_status_never_includes_message_text():
    events = _events(3, 's1', '+15550001', 1_700_000_000)
    report = status(events=events)
    blob = json.dumps(report).lower()
    assert 'hello there' not in blob
    assert '+15550001' not in blob


def test_status_counts_sources_and_reports_label_balance():
    events = _events(4, 's1', '+15550001', 1_700_000_000)
    report = status(events=events)
    assert report['group']['by_source'] == {'signal': 4}
    balance = report['label_balance']
    assert balance['responded'] + balance['silent'] == 4
    assert 0.0 <= balance['respond_rate'] <= 1.0


def test_summarise_states_not_ready_with_blockers():
    events = _events(5, 's1', '+15550001', 1_700_000_000)
    text = summarise(status(events=events))
    assert 'NOT READY' in text
    assert 'group_sessions' in text


def test_summarise_states_ready_when_gates_are_low():
    events = _events(6, 's1', '+15550001', 1_700_000_000)
    report = status(events=events, gates={'group_sessions': 1, 'group_decision_points': 5,
                                          'group_participants': 1})
    assert report['readiness']['ready'] is True
    assert 'READY' in summarise(report)


def test_real_corpus_report_is_wellformed():
    """Smoke test against whatever Hermes history exists on this machine."""
    report = status()
    assert report['group']['dataset']['decision_points'] >= 0
    assert isinstance(report['readiness']['ready'], bool)
    assert summarise(report)
