"""Habitat: simulation stepping, command queue, and live policy inference."""
import json

import numpy as np
import pytest

from notflyguy.chat_policy import ChatPolicy
from notflyguy.habitat import Habitat


class StubBrain:
    """Minimal stand-in for ConnectomeState so tests need no measured data."""

    def __init__(self, size=20):
        self.size = size
        self.activity = np.zeros(size)
        self.motor_indices = np.arange(5)
        self.upstream_indices = np.arange(5, 15)

    def reset(self):
        self.activity.fill(0.0)

    def step(self, drive):
        drive = np.asarray(drive, dtype=float)
        self.activity = np.tanh(0.2 * drive.sum() + 0.5 * self.activity)
        return self.activity

    @property
    def motor_activity(self):
        return self.activity[self.motor_indices]

    def features(self):
        a = self.activity
        return np.array([a.mean(), np.abs(a).mean(), np.sqrt((a ** 2).mean()),
                         self.motor_activity.mean(), np.abs(self.motor_activity).mean(),
                         a[self.upstream_indices].mean(),
                         np.abs(a[self.upstream_indices]).mean(), a.max()])

    @staticmethod
    def feature_names():
        return ('mean_activity', 'mean_abs_activity', 'activity_rms', 'mean_motor',
                'mean_abs_motor', 'mean_upstream', 'mean_abs_upstream', 'peak_activity')


def make_habitat(tmp_path, env=None, policy=True, brain=True):
    return Habitat(
        env=env,
        controller=None,
        chat_brain=StubBrain() if brain else None,
        policy=ChatPolicy(22) if policy else None,
        policy_path=None,
        shadow_log=tmp_path / 'shadow.jsonl',
        chat_activity=type('A', (), {'snapshot': staticmethod(lambda: {'phase': 'waiting'})})(),
    )


def test_snapshot_is_available_before_any_tick(tmp_path):
    state = make_habitat(tmp_path).snapshot()
    assert state['timestep'] == 0
    assert state['chat']['policy_loaded'] is True
    assert state['chat']['circuit_loaded'] is True


def test_tick_without_a_body_still_serves_state(tmp_path):
    habitat = make_habitat(tmp_path)
    state = habitat.tick()
    assert state['timestep'] == 0
    assert state['has_frame'] is False
    assert state['simulation_seconds'] is None


def test_tick_advances_a_body_and_records_reward(tmp_path, fake_env):
    habitat = make_habitat(tmp_path, env=fake_env)
    state = habitat.tick()
    assert state['timestep'] == 1
    assert state['reward'] == pytest.approx(0.25)
    assert fake_env.steps == 1


def test_episode_boundary_resets_the_body(tmp_path, fake_env):
    env = type(fake_env)(reward=1.0, last_after=2)
    habitat = make_habitat(tmp_path, env=env)
    habitat.tick()
    assert habitat.snapshot()['episode'] == 1
    state = habitat.tick()
    assert state['episode'] == 2
    assert env.resets == 1


def test_pause_resume_and_reset_commands(tmp_path, fake_env):
    habitat = make_habitat(tmp_path, env=fake_env)
    habitat.command('pause')
    assert habitat.tick()['paused'] is True
    habitat.command('resume')
    assert habitat.tick()['paused'] is False
    habitat.tick()
    assert habitat.snapshot()['timestep'] == 2
    habitat.command('reset')
    assert habitat.tick()['timestep'] == 1
    assert fake_env.resets == 1


def test_free_text_commands_become_the_goal(tmp_path):
    habitat = make_habitat(tmp_path)
    habitat.command('walk toward the light')
    assert habitat.tick()['goal'] == 'walk toward the light'


def test_group_message_produces_a_policy_decision(tmp_path):
    habitat = make_habitat(tmp_path)
    decision = habitat.observe_group_message(
        {'text': 'notflyguy, are you there?', 'sender': 'alice', 'sender_name': 'Alice'})
    assert decision['policy_loaded'] is True
    assert decision['addressed'] is True
    assert 0.0 <= decision['respond_probability'] <= 1.0
    assert set(decision['connectome']) == {'rms', 'peak', 'mean_motor', 'features'}
    assert len(decision['connectome']['features']) == 8
    assert decision['would_respond'] in (True, False)


def test_group_message_records_context_features(tmp_path):
    habitat = make_habitat(tmp_path)
    decision = habitat.observe_group_message({'text': 'hello there', 'sender': 'bob'})
    assert decision['context']['is_question'] == 0.0
    assert decision['context']['sender_is_new'] == 1.0
    habitat.observe_group_message({'text': 'are you awake?', 'sender': 'bob'})
    second = habitat.chat_last_decision
    assert second['context']['sender_is_new'] == 0.0
    assert second['context']['is_question'] == 1.0


def test_group_message_without_a_policy_reports_clearly(tmp_path):
    habitat = make_habitat(tmp_path, policy=False)
    decision = habitat.observe_group_message({'text': 'hi', 'sender': 'x'})
    assert decision['policy_loaded'] is False
    assert 'respond_probability' not in decision


def test_group_message_without_a_circuit_reports_an_error(tmp_path):
    habitat = make_habitat(tmp_path, brain=False)
    decision = habitat.observe_group_message({'text': 'hi', 'sender': 'x'})
    assert decision['error'] == 'connectome unavailable'


def test_agent_reply_advances_the_circuit_without_crashing(tmp_path):
    habitat = make_habitat(tmp_path)
    habitat.observe_group_message({'text': 'ping', 'sender': 'a'})
    before = habitat.chat_brain.activity.copy()
    habitat.observe_agent_reply('pong')
    assert not np.allclose(before, habitat.chat_brain.activity)


def test_sender_and_group_identifiers_are_not_stored_raw(tmp_path):
    habitat = make_habitat(tmp_path)
    decision = habitat.observe_group_message(
        {'text': 'hi', 'sender': '+15551234567', 'group': 'group:+15559999999'})
    assert '+15551234567' not in json.dumps(decision)
    assert '+15559999999' not in json.dumps(decision)


def test_shadow_log_records_predictions_without_message_text(tmp_path):
    habitat = make_habitat(tmp_path)
    habitat.observe_group_message({'text': 'my secret passphrase', 'sender': 'a'})
    rows = [json.loads(line) for line in (tmp_path / 'shadow.jsonl').read_text().splitlines()]
    assert rows and rows[0]['respond_probability'] is not None
    assert 'secret' not in json.dumps(rows).lower()
    assert 'text' not in rows[0]


def test_feed_is_bounded(tmp_path):
    habitat = make_habitat(tmp_path)
    for index in range(120):
        habitat.observe_group_message({'text': f'message {index}', 'sender': 'a'})
    assert len(habitat.chat_feed) <= 60
    assert len(habitat.snapshot()['chat']['feed']) <= 20


def test_counts_track_groups_and_messages(tmp_path):
    habitat = make_habitat(tmp_path)
    habitat.observe_group_message({'text': 'one', 'sender': 'a', 'group': 'g1'})
    habitat.observe_group_message({'text': 'two', 'sender': 'b', 'group': 'g2'})
    counts = habitat.snapshot()['chat']['counts']
    assert counts['groups'] == 2
    assert counts['messages'] == 2


def test_reports_reads_training_directories(tmp_path, monkeypatch):
    import notflyguy.habitat as habitat_module
    training = tmp_path / 'data' / 'training' / 'run-a'
    training.mkdir(parents=True)
    (training / 'report.json').write_text(json.dumps({'verdict': {'claim': 'x'}}))
    monkeypatch.setattr(habitat_module, 'ROOT', tmp_path)
    reports = make_habitat(tmp_path).reports()
    assert reports['run-a']['verdict']['claim'] == 'x'


def test_snapshot_is_replaced_not_mutated_in_place(tmp_path):
    habitat = make_habitat(tmp_path)
    first = habitat.snapshot()
    habitat.tick()
    assert habitat.snapshot() is not first


# --- real replies ------------------------------------------------------------
class FakeLLM:
    """Records the messages it was given and returns a canned reply."""

    model = 'fake-model'

    def __init__(self, reply='Bzz. Hello there!', error=None):
        self.reply = reply
        self.error = error
        self.calls: list[list[dict]] = []

    def complete(self, messages, max_tokens=200, **kwargs):
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        return self.reply


def make_talking_habitat(tmp_path, **kwargs):
    return Habitat(
        controller=None, chat_brain=StubBrain(), policy=ChatPolicy(22),
        policy_path=None, shadow_log=tmp_path / 'shadow.jsonl',
        chat_activity=type('A', (), {'snapshot': staticmethod(lambda: {'phase': 'waiting'})})(),
        **kwargs,
    )


def test_dashboard_chat_gets_a_real_reply(tmp_path):
    llm = FakeLLM(reply='Buzzing around the keys, doing great!')
    habitat = make_talking_habitat(tmp_path, llm=llm)
    decision = habitat.observe_group_message({'text': 'how are you?', 'sender': 'alice',
                                              'sender_name': 'Alice'})
    assert decision['reply'] == 'Buzzing around the keys, doing great!'
    assert llm.calls, 'the model should have been called'
    assert habitat.snapshot()['chat']['replies_enabled'] is True


def test_reply_prompt_carries_persona_and_policy_guidance(tmp_path):
    llm = FakeLLM()
    habitat = make_talking_habitat(tmp_path, llm=llm)
    habitat.observe_group_message({'text': 'notflyguy, thoughts?', 'sender': 'a'})
    system = llm.calls[0][0]
    assert system['role'] == 'system'
    assert 'NotFlyGuy' in system['content']
    assert 'advisory' in system['content'] or 'learned readout' in system['content']


def test_reply_prompt_includes_conversation_history(tmp_path):
    llm = FakeLLM()
    habitat = make_talking_habitat(tmp_path, llm=llm)
    habitat.observe_group_message({'text': 'first message', 'sender': 'alice',
                                   'sender_name': 'Alice'})
    habitat.observe_group_message({'text': 'second message', 'sender': 'bob',
                                   'sender_name': 'Bob'})
    second_call = llm.calls[-1]
    joined = ' '.join(m['content'] for m in second_call)
    assert 'first message' in joined
    assert 'second message' in joined


def test_signal_traffic_is_not_answered_here(tmp_path):
    """Hermes owns Signal replies; replying here would double-post."""
    llm = FakeLLM()
    habitat = make_talking_habitat(tmp_path, llm=llm)
    decision = habitat.observe_group_message({'text': 'hi from signal', 'sender': 'a',
                                              'source': 'signal'})
    assert 'reply' not in decision
    assert not llm.calls


def test_model_failure_is_reported_not_swallowed(tmp_path):
    llm = FakeLLM(error=RuntimeError('connection refused'))
    habitat = make_talking_habitat(tmp_path, llm=llm)
    decision = habitat.observe_group_message({'text': 'hello', 'sender': 'a'})
    assert 'reply' not in decision
    assert 'connection refused' in decision['reply_error']


def test_empty_model_reply_is_reported(tmp_path):
    habitat = make_talking_habitat(tmp_path, llm=FakeLLM(reply=''))
    decision = habitat.observe_group_message({'text': 'hello', 'sender': 'a'})
    assert decision['reply_error'] == 'model returned an empty reply'


def test_policy_gate_can_hold_a_reply_back(tmp_path):
    class SilentPolicy(ChatPolicy):
        def predict_proba(self, X):
            return np.zeros(len(X))

    llm = FakeLLM()
    habitat = Habitat(controller=None, chat_brain=StubBrain(), policy=SilentPolicy(22),
                      policy_path=None, shadow_log=None, llm=llm, policy_gate=True,
                      chat_activity=type('A', (), {
                          'snapshot': staticmethod(lambda: {'phase': 'waiting'})})())
    decision = habitat.observe_group_message({'text': 'hello', 'sender': 'a'})
    assert 'reply' not in decision
    assert 'policy gate' in decision['reply_skipped']
    assert not llm.calls


def test_policy_gate_is_off_by_default(tmp_path):
    """An unvalidated policy must not be allowed to ignore people silently."""
    class SilentPolicy(ChatPolicy):
        def predict_proba(self, X):
            return np.zeros(len(X))

    llm = FakeLLM(reply='still here')
    habitat = Habitat(controller=None, chat_brain=StubBrain(), policy=SilentPolicy(22),
                      policy_path=None, shadow_log=None, llm=llm,
                      chat_activity=type('A', (), {
                          'snapshot': staticmethod(lambda: {'phase': 'waiting'})})())
    decision = habitat.observe_group_message({'text': 'hello', 'sender': 'a'})
    assert decision.get('reply') == 'still here'


def test_reply_advances_the_circuit_and_history_stays_bounded(tmp_path):
    llm = FakeLLM()
    habitat = make_talking_habitat(tmp_path, llm=llm, history_limit=4)
    for index in range(12):
        habitat.observe_group_message({'text': f'message {index}', 'sender': 'a'})
    assert len(habitat.chat_history) <= 4
    assert habitat.chat_tracker.agent_replies == 12


def test_reply_is_not_generated_without_a_model(tmp_path):
    habitat = make_habitat(tmp_path)
    decision = habitat.observe_group_message({'text': 'hello', 'sender': 'a'})
    assert 'reply' not in decision
    assert habitat.snapshot()['chat']['replies_enabled'] is False
