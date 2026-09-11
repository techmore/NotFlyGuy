"""Hermes plugin: persona, telemetry gating, and the group-chat policy path."""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'fly_plugin', Path(__file__).parents[1] / 'integrations/hermes/notflyguy/__init__.py')
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test must opt in to a habitat call; nothing touches the network."""
    def fail(*args, **kwargs):
        raise AssertionError('unexpected habitat request')
    monkeypatch.setattr(plugin, '_request', fail)


def test_greeting_does_not_fetch_telemetry(monkeypatch):
    monkeypatch.setattr(plugin, 'request_fly', lambda *a, **k: pytest.fail('no telemetry for small talk'))
    monkeypatch.setattr(plugin, 'observe_group_message', lambda *a, **k: None)
    context = plugin.conversation_context("Hey fly, how's it going?", platform='signal')
    assert 'NotFlyGuy' in context
    assert 'Fresh technical data' not in context


def test_non_signal_platform_never_calls_the_habitat(monkeypatch):
    monkeypatch.setattr(plugin, 'observe_group_message',
                        lambda *a, **k: pytest.fail('must not call habitat off-Signal'))
    plugin.conversation_context('hello', platform='cli')


def test_technical_question_gets_live_data(monkeypatch):
    monkeypatch.setattr(plugin, 'request_fly', lambda command: 'live-test-data')
    monkeypatch.setattr(plugin, 'observe_group_message', lambda *a, **k: None)
    context = plugin.conversation_context('How many neurons are in your controller?', platform='signal')
    assert 'live-test-data' in context


def test_policy_decision_is_injected_as_advisory_guidance(monkeypatch):
    decision = {'policy_loaded': True, 'respond_probability': 0.12,
                'predicted_reply_length': 0.2, 'connectome': {'rms': 0.31}}
    monkeypatch.setattr(plugin, 'observe_group_message', lambda *a, **k: decision)
    context = plugin.conversation_context('anyone about?', platform='signal', session_id='s1')
    assert 'p=0.12' in context
    assert 'hold back' in context
    assert 'advisory only' in context
    assert 'must still answer' in context


def test_style_guidance_is_empty_without_a_policy():
    assert plugin.style_guidance(None, 'group') == ''
    assert plugin.style_guidance({'policy_loaded': False}, 'group') == ''
    assert plugin.style_guidance({'policy_loaded': True}, 'group') == ''


def test_style_guidance_requests_a_short_reply_for_low_length():
    decision = {'policy_loaded': True, 'respond_probability': 0.9, 'predicted_reply_length': 0.1,
                'connectome': {'rms': 0.0}}
    guidance = plugin.style_guidance(decision, 'group')
    assert 'speak here' in guidance
    assert 'sentence or two' in guidance


def test_guidance_never_tells_the_model_to_stay_silent():
    """Hermes cannot suppress a send from pre_llm_call, so guidance must not pretend to."""
    decision = {'policy_loaded': True, 'respond_probability': 0.01, 'predicted_reply_length': 0.1,
                'connectome': {'rms': 0.0}}
    guidance = plugin.style_guidance(decision, 'group')
    assert 'do not reply' not in guidance.lower()
    assert 'stay silent' not in guidance.lower()


def test_session_chat_type_reads_the_database(tmp_path, monkeypatch):
    path = tmp_path / 'state.db'
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, chat_type TEXT)')
    connection.execute("INSERT INTO sessions VALUES ('g1', 'group')")
    connection.execute("INSERT INTO sessions VALUES ('d1', NULL)")
    connection.commit()
    connection.close()
    monkeypatch.setenv('NOTFLYGUY_HERMES_DB', str(path))
    assert plugin.session_chat_type('g1') == 'group'
    assert plugin.session_chat_type('missing') == ''
    assert plugin.session_chat_type('') == ''


def test_session_chat_type_survives_a_missing_database(tmp_path, monkeypatch):
    monkeypatch.setenv('NOTFLYGUY_HERMES_DB', str(tmp_path / 'absent.db'))
    assert plugin.session_chat_type('anything') == ''


def test_before_reply_returns_context_and_publishes(monkeypatch):
    published = []
    monkeypatch.setattr(plugin, 'publish_chat', lambda phase, text: published.append(phase))
    monkeypatch.setattr(plugin, 'session_chat_type', lambda session_id: 'group')
    monkeypatch.setattr(plugin, 'observe_group_message',
                        lambda *a, **k: {'policy_loaded': False})
    result = plugin.before_reply(platform='signal', user_message='hi', session_id='s1')
    assert 'context' in result
    assert published == ['typing']


def test_before_reply_ignores_other_platforms(monkeypatch):
    monkeypatch.setattr(plugin, 'publish_chat',
                        lambda *a: pytest.fail('must not publish off-Signal'))
    assert plugin.before_reply(platform='cli', user_message='hi') is None


def test_after_reply_publishes_and_feeds_the_circuit(monkeypatch):
    seen = {}
    monkeypatch.setattr(plugin, 'publish_chat', lambda phase, text: seen.setdefault('phase', phase))
    monkeypatch.setattr(plugin, 'observe_agent_reply', lambda text: seen.setdefault('reply', text))
    plugin.after_reply(platform='signal', assistant_response='buzz')
    assert seen == {'phase': 'ready', 'reply': 'buzz'}


def test_habitat_tool_rejects_unsupported_commands():
    import json
    assert 'error' in json.loads(plugin.habitat_tool({'command': 'explode'}))


def test_request_fly_help_text_mentions_the_viewer():
    text = plugin.request_fly('help')
    assert '/fly' in text and plugin.BASE in text


def test_request_fly_reports_an_unavailable_habitat(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError('connection refused')
    monkeypatch.setattr(plugin, '_request', boom)
    assert 'unavailable' in plugin.request_fly('status')
