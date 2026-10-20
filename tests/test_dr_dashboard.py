"""DR adapter dashboard: origin defence for the command-sending endpoints."""
import json
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dr_adapter.dashboard import Adapter, Handler, origin_ok
from dr_adapter.transport import MockTransport


@pytest.fixture
def dr_server():
    class TestHandler(Handler):
        adapter = Adapter(MockTransport())
        training_state = None

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), TestHandler)
    import threading
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address[:2]
    try:
        yield f'http://{host}:{port}'
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url, payload, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read() or b'null')


def test_origin_ok_is_same_origin_only():
    assert origin_ok({'Host': '127.0.0.1:8787'}) is True                       # non-browser
    assert origin_ok({'Host': '127.0.0.1:8787', 'Origin': 'http://127.0.0.1:8787'}) is True
    assert origin_ok({'Host': '127.0.0.1:8787', 'Origin': 'http://127.0.0.1:1'}) is False
    assert origin_ok({'Host': '127.0.0.1:8787', 'Origin': 'http://evil.example'}) is False
    assert origin_ok({'Host': '127.0.0.1:8787', 'Origin': 'null'}) is False
    assert origin_ok({'Origin': 'http://127.0.0.1:8787'}) is False              # no Host


def test_cross_origin_approve_is_refused(dr_server):
    status, body = _post(dr_server + '/api/approve', {}, origin='http://evil.example')
    assert status == 403
    assert body == {'error': 'origin not allowed'}


def test_same_origin_and_non_browser_writes_still_work(dr_server):
    status, body = _post(dr_server + '/api/pause', {'paused': True}, origin=dr_server)
    assert status == 200 and body['paused'] is True
    status, body = _post(dr_server + '/api/pause', {'paused': False})
    assert status == 200 and body['paused'] is False


def test_state_endpoint_is_readable(dr_server):
    with urlopen(dr_server + '/api/state', timeout=5) as response:
        snapshot = json.loads(response.read())
    assert snapshot['armed'] is True
    assert snapshot['proposal'] is None
