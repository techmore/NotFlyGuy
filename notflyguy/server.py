"""HTTP adapter over :mod:`notflyguy.habitat`.

Endpoints
---------
``GET  /``                dashboard page
``GET  /events``          Server-sent events with JSON state (no image payload)
``GET  /frame.jpg``       the current camera frame as JPEG
``GET  /api/state``       one JSON snapshot
``POST /api/signal``      queue a habitat command (pause/resume/reset/goal)
``POST /api/chat-events`` dashboard chat feed from the Hermes plugin
``POST /api/group-message`` one group-chat message -> connectome + policy decision
``POST /api/agent-reply``   the agent spoke, so the circuit can register closure
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .habitat import Habitat

WEB = Path(__file__).parent / 'web'
HOST = '127.0.0.1'
PORT = 8765
MAX_BODY = 16384
EVENT_INTERVAL = 0.1

HABITAT: Habitat | None = None


def _json_response(handler, value, status=200):
    payload = json.dumps(value).encode()
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Content-Length', str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _error(handler, status, message):
    _json_response(handler, {'error': message}, status=status)


def _read_json(handler, limit=MAX_BODY):
    """Return (body, None) or (None, error_message)."""
    try:
        length = int(handler.headers.get('Content-Length', '0'))
    except (TypeError, ValueError):
        return None, 'invalid length'
    if not 0 < length <= limit:
        return None, 'body too large or empty'
    try:
        return json.loads(handler.rfile.read(length)), None
    except (ValueError, UnicodeError):
        return None, 'invalid JSON'


def _origin_ok(handler) -> bool:
    """Same-origin check for browser writes; non-browser clients send no Origin.

    Comparing against the request's own Host header (rather than a hardcoded
    port) keeps this correct on any port and still blocks a page served from a
    different origin from driving the habitat.
    """
    origin = handler.headers.get('Origin')
    if origin is None:
        return True
    host = handler.headers.get('Host')
    if not host:
        return False
    parsed = urlsplit(origin)
    return parsed.scheme in ('http', 'https') and parsed.netloc == host


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'NotFlyGuy/0.2'

    def log_message(self, *args):
        pass

    # -- helpers -----------------------------------------------------------
    def _habitat(self) -> Habitat | None:
        return HABITAT

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        habitat = self._habitat()
        route = urlsplit(self.path).path
        if route == '/api/state':
            if habitat is None:
                return _error(self, 503, 'habitat unavailable')
            return _json_response(self, habitat.snapshot())
        if route == '/frame.jpg':
            if habitat is None or habitat.frame_jpeg is None:
                return self.send_error(404)
            image = habitat.frame_jpeg
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(image)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return self.wfile.write(image)
        if route == '/api/report':
            if habitat is None:
                return _error(self, 503, 'habitat unavailable')
            return _json_response(self, habitat.reports())
        if route == '/api/corpus':
            if habitat is None:
                return _error(self, 503, 'habitat unavailable')
            return _json_response(self, habitat.corpus_report())
        if route == '/events':
            return self._serve_events(habitat)
        return self._serve_static(route)

    def _serve_events(self, habitat):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            while True:
                if habitat is None:
                    payload = json.dumps({'error': 'habitat unavailable'})
                else:
                    payload = json.dumps(habitat.snapshot())
                self.wfile.write(f'data: {payload}\n\n'.encode())
                self.wfile.flush()
                time.sleep(EVENT_INTERVAL)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _serve_static(self, route):
        relative = 'index.html' if route in ('/', '') else route.lstrip('/')
        candidate = (WEB / relative).resolve()
        if candidate.is_file() and WEB.resolve() in candidate.parents:
            body = candidate.read_bytes()
            content_type = ('text/html; charset=utf-8' if candidate.suffix == '.html'
                            else 'text/plain; charset=utf-8')
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        return self.send_error(404)

    # -- POST --------------------------------------------------------------
    def do_POST(self):
        if not _origin_ok(self):
            return _error(self, 403, 'origin not allowed')
        habitat = self._habitat()
        if habitat is None:
            return _error(self, 503, 'habitat unavailable')
        route = urlsplit(self.path).path
        if route == '/api/chat-events':
            return self._post_chat_event(habitat)
        if route == '/api/signal':
            return self._post_signal(habitat)
        if route == '/api/group-message':
            return self._post_group_message(habitat)
        if route == '/api/agent-reply':
            return self._post_agent_reply(habitat)
        return _error(self, 404, 'unknown endpoint')

    def _post_chat_event(self, habitat):
        body, error = _read_json(self)
        if error:
            return _error(self, 400, error)
        if (not isinstance(body, dict) or body.get('phase') not in ('typing', 'ready')
                or not isinstance(body.get('text'), str) or len(body['text']) > 6000):
            return _error(self, 400, 'expected {phase: typing|ready, text: str<=6000}')
        habitat.publish_chat(body['phase'], body['text'])
        return _json_response(self, {'ok': True})

    def _post_signal(self, habitat):
        body, error = _read_json(self, limit=4096)
        if error:
            return _error(self, 400, error)
        if not isinstance(body, dict) or not isinstance(body.get('message'), str):
            return _error(self, 400, 'expected {message: str}')
        message = body['message'].strip()
        if not message:
            return _error(self, 400, 'empty message')
        result = habitat.command(message)
        return _json_response(self, {'ok': True, **result})

    def _post_group_message(self, habitat):
        body, error = _read_json(self)
        if error:
            return _error(self, 400, error)
        if not isinstance(body, dict) or not isinstance(body.get('text'), str):
            return _error(self, 400, 'expected {text: str}')
        if not body['text'].strip():
            return _error(self, 400, 'empty text')
        if len(body['text']) > 2000:
            return _error(self, 400, 'text too long')
        decision = habitat.observe_group_message(body)
        return _json_response(self, {'ok': True, 'decision': decision})

    def _post_agent_reply(self, habitat):
        body, error = _read_json(self, limit=4096)
        if error:
            return _error(self, 400, error)
        if not isinstance(body, dict):
            return _error(self, 400, 'expected an object')
        habitat.observe_agent_reply(str(body.get('text', '')))
        return _json_response(self, {'ok': True})


def serve(habitat: Habitat, host: str = HOST, port: int = PORT, block: bool = True,
          tick_interval: float = 0.1):
    """Start the HTTP server and the simulation loop."""
    global HABITAT
    HABITAT = habitat
    httpd = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f'NotFlyGuy live at http://{host}:{port}')
    if not block:
        return httpd
    try:
        while True:
            habitat.tick()
            time.sleep(tick_interval)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return httpd


def main():
    import os
    parser = argparse.ArgumentParser(description='Run the NotFlyGuy habitat and dashboard.')
    parser.add_argument('--host', default=HOST)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--no-body', action='store_true',
                        help='skip loading FlyBody (chat and circuit only)')
    parser.add_argument('--policy', default=None, help='path to a trained chat policy .npz')
    parser.add_argument('--llm-url', default=os.environ.get('NOTFLYGUY_LLM_URL',
                                                            'http://127.0.0.1:8083'),
                        help='OpenAI-compatible endpoint used to answer dashboard chat')
    parser.add_argument('--llm-model', default=os.environ.get('NOTFLYGUY_LLM_MODEL'))
    parser.add_argument('--no-replies', action='store_true',
                        help='monitor only; do not generate replies in dashboard chat')
    parser.add_argument('--policy-gate', action='store_true',
                        help='let the (unvalidated) policy suppress replies; off by default')
    args = parser.parse_args()
    from .habitat import DEFAULT_POLICY
    habitat = Habitat.create(policy_path=args.policy or DEFAULT_POLICY,
                             with_body=not args.no_body,
                             llm_url=None if args.no_replies else args.llm_url,
                             llm_model=args.llm_model,
                             policy_gate=args.policy_gate)
    chat = habitat.snapshot()['chat']
    if chat['replies_enabled']:
        print(f"Replies enabled via {args.llm_url} (model {chat['reply_model']})")
    else:
        print('Replies disabled: no language model reachable; monitoring only')
    serve(habitat, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
