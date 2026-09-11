"""Transcript loading: identity handling, filtering, and text-free summaries."""
import json

from notflyguy.chatlog import ChatEvent, corpus_stats, load_events, load_all, pseudonym


def test_pseudonym_is_stable_distinct_and_not_reversible():
    assert pseudonym('+15550001') == pseudonym('+15550001')
    assert pseudonym('+15550001') != pseudonym('+15550002')
    assert '+15550001' not in pseudonym('+15550001')
    assert len(pseudonym('x')) == 12
    assert pseudonym(None) == 'unknown'
    assert pseudonym('+15550001', salt='other') != pseudonym('+15550001')


def test_group_loading_skips_empty_and_compacted_rows(hermes_db):
    events = load_events(hermes_db, chat_types=('group',))
    assert len(events) == 9
    assert sum(1 for e in events if e.is_agent) == 4
    assert sum(1 for e in events if not e.is_agent) == 5
    assert all(e.chat_type == 'group' for e in events)
    assert all(e.source == 'signal' for e in events)


def test_chat_type_filter_excludes_direct_sessions(hermes_db):
    groups = load_events(hermes_db, chat_types=('group',))
    everything = load_events(hermes_db, chat_types=None)
    assert len(everything) > len(groups)
    assert all(e.chat_type != 'group' for e in everything if e.session_id == 's-direct-1')


def test_public_view_omits_message_text(hermes_db):
    events = load_events(hermes_db)
    for event in events:
        public = event.to_public()
        assert 'text' not in public
        assert public['text_length'] == len(event.text)
        assert 'buzzing' not in json.dumps(public).lower()


def test_participant_identity_is_pseudonymised(hermes_db):
    events = load_events(hermes_db)
    serialised = json.dumps([e.to_public() for e in events])
    assert '+15550001' not in serialised
    assert '+15550009' not in serialised
    owners = [e for e in events if e.sender_name == 'Owner']
    assert owners, 'session owner should be named from origin_json'
    assert all(e.sender_id != 'agent' for e in owners)


def test_addressed_detection(hermes_db):
    events = load_events(hermes_db)
    first = events[0]
    assert first.addressed_to_agent is True
    assert first.text.lower().startswith('notflyguy')


def test_corpus_stats_contain_no_text(hermes_db):
    events = load_events(hermes_db)
    stats = corpus_stats(events)
    assert stats['events'] == 9
    assert stats['sessions'] == 2
    assert stats['groups'] == 2
    assert stats['user_messages'] == 5
    assert 'notflyguy' not in json.dumps(stats).lower()


def test_load_all_tolerates_missing_database(tmp_path, hermes_db):
    events = load_all([tmp_path / 'does-not-exist.db', hermes_db], chat_types=('group',))
    assert len(events) == 9


def test_events_are_ordered_within_a_session(hermes_db):
    events = load_events(hermes_db)
    for session in {e.session_id for e in events}:
        rows = [e.timestamp for e in events if e.session_id == session]
        assert rows == sorted(rows)


def test_chat_event_is_hashable_and_frozen(group_events):
    event = group_events[0]
    assert isinstance(event, ChatEvent)
    assert hash(event) == hash(event)
    try:
        event.text = 'mutated'
    except Exception:
        return
    raise AssertionError('ChatEvent should be immutable')
