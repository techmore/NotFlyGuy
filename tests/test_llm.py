"""OpenAI-compatible client: discovery, reasoning models, and failure modes."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from notflyguy.llm import LLMUnavailable, LocalLLM


class _Stub(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    model_ids = ['stub-model']
    completion = {}
    reject_thinking = False
    seen_payloads = []

    def log_message(self, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/v1/models':
            return self._send({'object': 'list',
                               'data': [{'id': m} for m in type(self).model_ids]})
        return self._send({'error': 'not found'}, status=404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(length)) if length else {}
        type(self).seen_payloads.append(payload)
        if self.path != '/v1/chat/completions':
            return self._send({'error': 'not found'}, status=404)
        if type(self).reject_thinking and payload.get('chat_template_kwargs'):
            return self._send({'error': 'unknown parameter'}, status=400)
        return self._send(type(self).completion)


@pytest.fixture
def stub_server():
    _Stub.model_ids = ['stub-model']
    _Stub.completion = {'choices': [{'finish_reason': 'stop',
                                     'message': {'role': 'assistant', 'content': 'hello there'}}]}
    _Stub.reject_thinking = False
    _Stub.seen_payloads = []
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Stub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f'http://{host}:{port}', _Stub
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_models_and_auto_resolution(stub_server):
    url, _ = stub_server
    client = LocalLLM(url)
    assert client.models() == ['stub-model']
    assert client.resolve_model() == 'stub-model'
    assert client.available() is True


def test_complete_returns_content(stub_server):
    url, stub = stub_server
    client = LocalLLM(url)
    assert client.complete([{'role': 'user', 'content': 'hi'}]) == 'hello there'
    assert stub.seen_payloads[-1]['model'] == 'stub-model'


def test_thinking_is_disabled_by_default(stub_server):
    url, stub = stub_server
    LocalLLM(url).complete([{'role': 'user', 'content': 'hi'}])
    assert stub.seen_payloads[-1]['chat_template_kwargs'] == {'enable_thinking': False}


def test_client_retries_without_thinking_when_rejected(stub_server):
    url, stub = stub_server
    stub.reject_thinking = True
    client = LocalLLM(url)
    assert client.complete([{'role': 'user', 'content': 'hi'}]) == 'hello there'
    assert client._thinking_supported is False
    assert stub.seen_payloads[0].get('chat_template_kwargs') is not None
    assert 'chat_template_kwargs' not in stub.seen_payloads[1]

    # And it should not keep retrying on later calls.
    before = len(stub.seen_payloads)
    client.complete([{'role': 'user', 'content': 'again'}])
    assert len(stub.seen_payloads) == before + 1


def test_reasoning_that_consumes_the_budget_is_reported(stub_server):
    url, stub = stub_server
    stub.completion = {'choices': [{
        'finish_reason': 'length',
        'message': {'role': 'assistant', 'reasoning': 'thinking...', 'content': None},
    }]}
    with pytest.raises(LLMUnavailable) as info:
        LocalLLM(url).complete([{'role': 'user', 'content': 'hi'}])
    assert 'hidden reasoning' in str(info.value)


def test_empty_content_returns_empty_string(stub_server):
    url, stub = stub_server
    stub.completion = {'choices': [{'finish_reason': 'stop',
                                    'message': {'role': 'assistant', 'content': '   '}}]}
    assert LocalLLM(url).complete([{'role': 'user', 'content': 'hi'}]) == ''


def test_malformed_completion_shape_raises(stub_server):
    url, stub = stub_server
    stub.completion = {'unexpected': True}
    with pytest.raises(LLMUnavailable):
        LocalLLM(url).complete([{'role': 'user', 'content': 'hi'}])


def test_unreachable_endpoint_is_unavailable():
    client = LocalLLM('http://127.0.0.1:1', timeout=1.0)
    assert client.available() is False
    with pytest.raises(LLMUnavailable):
        client.complete([{'role': 'user', 'content': 'hi'}])


def test_no_models_advertised_is_unavailable(stub_server):
    url, stub = stub_server
    stub.model_ids = []
    assert LocalLLM(url).available() is False


def test_explicit_model_is_not_overridden(stub_server):
    url, stub = stub_server
    stub.model_ids = ['a', 'b']
    client = LocalLLM(url, model='b')
    client.complete([{'role': 'user', 'content': 'hi'}])
    assert stub.seen_payloads[-1]['model'] == 'b'
