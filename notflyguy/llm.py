"""Minimal OpenAI-compatible client, standard library only.

Deliberately dependency-free: the dashboard already runs without a GPU stack,
and adding an HTTP client library for one endpoint is not worth the packaging
cost. Talks to any server exposing ``/v1/models`` and
``/v1/chat/completions`` -- LM Studio, llama.cpp's server, Ollama's compat
endpoint, or a hosted API with a base URL and key.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = 'http://127.0.0.1:8083'
DEFAULT_TIMEOUT = 60.0


class LLMUnavailable(RuntimeError):
    """The model endpoint could not be reached or returned an error."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class LocalLLM:
    """Chat completions against an OpenAI-compatible endpoint.

    Reasoning models are common on local servers, and a thinking model will
    happily spend the entire token budget on hidden reasoning and return no
    answer at all. ``disable_thinking`` asks the server to skip that (LM Studio
    and several others accept ``chat_template_kwargs.enable_thinking``); on a
    server that rejects the parameter the client retries without it and
    remembers, rather than failing.
    """

    def __init__(self, base_url=DEFAULT_BASE_URL, model=None, api_key=None,
                 timeout=DEFAULT_TIMEOUT, disable_thinking=True):
        self.base_url = str(base_url).rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.disable_thinking = bool(disable_thinking)
        self._thinking_supported = bool(disable_thinking)

    # -- discovery ---------------------------------------------------------
    def _request(self, path, payload=None, timeout=None):
        url = self.base_url + path
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(url, data=data, headers=headers,
                          method='POST' if data is not None else 'GET')
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise LLMUnavailable(f'HTTP {error.code} from {url}', status=error.code) from error
        except (URLError, TimeoutError, OSError) as error:
            raise LLMUnavailable(f'{type(error).__name__} contacting {url}: {error}') from error
        except ValueError as error:
            raise LLMUnavailable(f'malformed response from {url}: {error}') from error

    def models(self) -> list[str]:
        payload = self._request('/v1/models', timeout=5.0)
        return [item.get('id', '') for item in payload.get('data', []) if item.get('id')]

    def resolve_model(self) -> str:
        """Return the configured model, or the first one the server advertises."""
        if self.model:
            return self.model
        available = self.models()
        if not available:
            raise LLMUnavailable(f'{self.base_url} advertises no models')
        self.model = available[0]
        return self.model

    def available(self) -> bool:
        try:
            self.resolve_model()
            return True
        except LLMUnavailable:
            return False

    # -- generation --------------------------------------------------------
    def _payload(self, messages, max_tokens, temperature, top_p, stop, disable_thinking):
        payload = {'model': self.resolve_model(), 'messages': list(messages),
                   'max_tokens': int(max_tokens), 'temperature': float(temperature),
                   'top_p': float(top_p), 'stream': False}
        if stop:
            payload['stop'] = list(stop)
        if disable_thinking:
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        return payload

    def complete(self, messages, max_tokens=220, temperature=0.7, top_p=0.95,
                 stop=None, disable_thinking=None) -> str:
        """Return the assistant text. Raises :class:`LLMUnavailable` on failure."""
        use_disable = (self.disable_thinking if disable_thinking is None
                       else bool(disable_thinking)) and self._thinking_supported
        payload = self._payload(messages, max_tokens, temperature, top_p, stop, use_disable)
        try:
            response = self._request('/v1/chat/completions', payload)
        except LLMUnavailable as error:
            # A server that does not understand the parameter may reject it.
            if not use_disable or error.status not in (400, 404, 422):
                raise
            self._thinking_supported = False
            payload = self._payload(messages, max_tokens, temperature, top_p, stop, False)
            response = self._request('/v1/chat/completions', payload)

        try:
            choice = response['choices'][0]
            message = choice.get('message') or {}
        except (KeyError, IndexError, TypeError) as error:
            raise LLMUnavailable(f'unexpected completion shape: {error}') from error

        content = (message.get('content') or '').strip()
        if content:
            return content
        # Some reasoning models return only hidden reasoning when the budget runs
        # out. Say so instead of silently returning an empty reply.
        reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
        if choice.get('finish_reason') == 'length' and reasoning:
            raise LLMUnavailable(
                'model spent the whole token budget on hidden reasoning and returned no '
                'answer; raise max_tokens')
        return ''
