from notflyguy.chat_activity import ChatActivity


def test_signal_activity_excludes_content_and_cli(monkeypatch, tmp_path):
    log = tmp_path / 'agent.log'
    log.write_text('')
    monkeypatch.setenv('NOTFLYGUY_GATEWAY_LOG', str(log))
    activity = ChatActivity()
    activity.consume('conversation turn: session=cli platform=cli msg=private')
    assert activity.state['phase'] == 'waiting'
    activity.consume('conversation turn: session=signal platform=signal msg=private')
    assert activity.state['phase'] == 'typing'
    assert 'private' not in str(activity.state)
    activity.consume('[signal] tool notflyguy_habitat completed')
    assert activity.state['phase'] == 'typing'
    activity.consume('[signal] Turn ended:')
    assert activity.state['phase'] == 'ready'
    activity.consume('gateway.platforms.signal: [Signal] Sending response chunk 1/1')
    assert activity.state['phase'] == 'handoff'
