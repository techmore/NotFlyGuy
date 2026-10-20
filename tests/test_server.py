"""HTTP adapter: routing, limits, origin checks, and traversal defence."""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from notflyguy import server as server_module
from notflyguy.chat_policy import ChatPolicy
from notflyguy.habitat import Habitat


class StubBrain:
    def __init__(self):
        import numpy as np
        self.activity = np.zeros(16)
        self.motor_indices = np.arange(4)
        self.upstream_indices = np.arange(4, 12)

    def reset(self):
        import numpy as np
        self.activity = np.zeros(16)

    def step(self, drive):
        import numpy as np
        self.activity = np.tanh(0.1 * np.asarray(drive).sum() + 0.5 * self.activity)
        return self.activity

    @property
    def motor_activity(self):
        return self.activity[self.motor_indices]

    def features(self):
        import numpy as np
        return np.zeros(8)

    @staticmethod
    def feature_names():
        return tuple(f'f{i}' for i in range(8))


@pytest.fixture
def live_server(tmp_path):
    habitat = Habitat(
        chat_brain=StubBrain(),
        policy=ChatPolicy(22),
        policy_path=None,
        shadow_log=tmp_path / 'shadow.jsonl',
        chat_activity=type('A', (), {'snapshot': staticmethod(lambda: {'phase': 'waiting'})})(),
    )
    httpd = server_module.serve(habitat, port=0, block=False)
    host, port = httpd.server_address[:2]
    base = f'http://{host}:{port}'
    try:
        yield base, habitat
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url, **kwargs):
    with urlopen(url, timeout=5, **kwargs) as response:
        return response.status, response.read(), response.headers


def _post(url, payload, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, None


def test_serves_the_dashboard(live_server):
    base, _ = live_server
    status, body, headers = _get(base + '/')
    assert status == 200
    assert b'NotFlyGuy' in body
    assert 'text/html' in headers['Content-Type']


def test_state_endpoint_returns_json(live_server):
    base, _ = live_server
    status, body, headers = _get(base + '/api/state')
    assert status == 200
    assert 'application/json' in headers['Content-Type']
    assert json.loads(body)['timestep'] == 0


def test_unknown_path_is_404(live_server):
    base, _ = live_server
    with pytest.raises(HTTPError) as info:
        _get(base + '/nope')
    assert info.value.code == 404


def test_path_traversal_is_refused(live_server):
    base, _ = live_server
    with pytest.raises(HTTPError) as info:
        _get(base + '/../server.py')
    assert info.value.code == 404


def test_report_endpoint_returns_a_mapping(live_server):
    base, _ = live_server
    status, body, _ = _get(base + '/api/report')
    assert status == 200
    assert isinstance(json.loads(body), dict)


def test_group_message_endpoint_returns_a_decision(live_server):
    base, _ = live_server
    status, body = _post(base + '/api/group-message', {'text': 'hello fly', 'sender': 'alice'})
    assert status == 200
    assert body['ok'] is True
    assert 'respond_probability' in body['decision']


def test_group_message_rejects_empty_and_oversized_text(live_server):
    base, _ = live_server
    assert _post(base + '/api/group-message', {'text': '   '})[0] == 400
    assert _post(base + '/api/group-message', {'text': 'x' * 3000})[0] == 400
    assert _post(base + '/api/group-message', {'nope': 1})[0] == 400


def test_cross_origin_write_is_refused(live_server):
    base, _ = live_server
    assert _post(base + '/api/signal', {'message': 'pause'}, origin='http://evil.example')[0] == 403
    # A page on a different local port is a different origin too.
    assert _post(base + '/api/signal', {'message': 'pause'},
                 origin='http://127.0.0.1:1')[0] == 403


def test_same_origin_write_is_allowed(live_server):
    base, _ = live_server
    status, body = _post(base + '/api/signal', {'message': 'pause'}, origin=base)
    assert status == 200
    assert body['queued'] is True


def test_signal_endpoint_validates_the_payload(live_server):
    base, _ = live_server
    assert _post(base + '/api/signal', {'message': ''})[0] == 400
    assert _post(base + '/api/signal', {'message': 5})[0] == 400
    assert _post(base + '/api/signal', {'message': 'x' * 5000})[0] == 400


def test_chat_event_endpoint_validates_phase(live_server):
    base, _ = live_server
    assert _post(base + '/api/chat-events', {'phase': 'typing', 'text': 'hi'})[0] == 200
    assert _post(base + '/api/chat-events', {'phase': 'bogus', 'text': 'hi'})[0] == 400


def test_agent_reply_endpoint(live_server):
    base, _ = live_server
    assert _post(base + '/api/agent-reply', {'text': 'pong'})[0] == 200


def test_frame_endpoint_is_404_without_a_body(live_server):
    base, _ = live_server
    with pytest.raises(HTTPError) as info:
        _get(base + '/frame.jpg')
    assert info.value.code == 404


def test_query_strings_do_not_break_routing(live_server):
    """The GUI cache-busts the frame URL, so a query string must still route."""
    base, habitat = live_server
    habitat.frame_jpeg = b'\xff\xd8\xff\xd9'  # minimal JPEG marker pair
    status, body, headers = _get(base + '/frame.jpg?t=123456')
    assert status == 200
    assert body == b'\xff\xd8\xff\xd9'
    assert headers['Content-Type'] == 'image/jpeg'
    assert json.loads(_get(base + '/api/state?cache=0')[1])['timestep'] == 0


def test_events_stream_is_not_inline_framed(live_server):
    """The frame must not ride the SSE payload; it is fetched separately."""
    base, _ = live_server
    request = Request(base + '/events')
    with urlopen(request, timeout=5) as response:
        chunk = response.read(4096)
    assert b'data: ' in chunk
    assert b'data:image/jpeg;base64' not in chunk


def test_handler_returns_503_without_a_habitat(live_server, monkeypatch):
    base, _ = live_server
    monkeypatch.setattr(server_module, 'HABITAT', None)
    for route in ('/api/state', '/api/report', '/api/corpus'):
        with pytest.raises(HTTPError) as info:
            _get(base + route)
        assert info.value.code == 503, route
        assert json.loads(info.value.read()) == {'error': 'habitat unavailable'}
    assert _post(base + '/api/signal', {'message': 'pause'})[0] == 503
    # The page itself still serves, so the dashboard can show its own error state.
    assert _get(base + '/')[0] == 200
