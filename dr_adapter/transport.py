from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen

from .action_space import Action
from .observation import parse_transcript


@dataclass(frozen=True)
class GameEvent:
    text: str
    category: str = "screen"

@dataclass(frozen=True)
class SendResult:
    ok: bool
    message: str
    messages: tuple[str, ...] = ()

class GameTransport(Protocol):
    def read_events(self) -> list[GameEvent]: ...
    def propose(self, action: Action) -> None: ...
    def send_approved(self, action: Action) -> SendResult: ...
    def stop(self) -> None: ...

class MockTransport:
    def __init__(self, transcript: str = ""):
        self.events = [GameEvent(transcript)] if transcript else []
        self.proposals: list[Action] = []
        self.sent: list[Action] = []
        self.stopped = False
    def read_events(self): return list(self.events)
    def propose(self, action): self.proposals.append(action)
    def send_approved(self, action):
        if self.stopped: return SendResult(False, "transport stopped")
        self.sent.append(action); return SendResult(True, "mock send")
    def stop(self): self.stopped = True

class DragonRealmsHTTPTransport:
    """Real local DR API transport. Credentials/tokens are supplied by caller."""
    def __init__(self, base_url: str, token: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.stopped = False
        self.last_messages: tuple[dict, ...] = ()
        self.requirements: list[dict] | None = None
    @classmethod
    def bootstrap(cls, base_url, user, password, character, race='gortog', guild='barbarian', timeout=3.0):
        """Create a disposable account/character and return an entered transport."""
        base = base_url.rstrip('/')
        def call(path, method='POST', payload=None, token=None):
            headers={'Content-Type':'application/json'}
            if token: headers['Authorization']='Bearer '+token
            data=json.dumps(payload or {}).encode()
            return json.loads(urlopen(Request(base+path, data=data, headers=headers, method=method), timeout=timeout).read())
        reg=call('/api/register', payload={'user':user,'pass':password})
        if not reg.get('ok'):
            reg=call('/api/login', payload={'user':user,'pass':password})
        if not reg.get('ok'): raise RuntimeError(reg.get('error','login failed'))
        token=reg['token']; chars=call('/api/characters','GET',token=token).get('characters',[])
        row=next((c for c in chars if c.get('name') == character), None)
        if row is None:
            created=call('/api/characters',payload={'name':character,'race':race,'guild':guild},token=token)
            if not created.get('ok'): raise RuntimeError(created.get('error','character creation failed'))
            char_id=created['charId']
        else: char_id=row.get('id', row.get('charId'))
        entered=call('/api/enter',payload={'charId':char_id},token=token)
        if not entered.get('ok'): raise RuntimeError(entered.get('error','enter failed'))
        transport=cls(base_url, token, timeout)
        for msg in entered.get('messages', []):
            if isinstance(msg, dict) and isinstance(msg.get('requirements'), list): transport.requirements=msg['requirements']
        transport.read_state(); return transport
    def _request(self, path, method='GET', payload=None):
        headers = {'Authorization': f'Bearer {self.token}'}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
        response = urlopen(Request(self.base_url + path, data=data, headers=headers, method=method), timeout=self.timeout)
        return json.loads(response.read())
    def read_events(self):
        if self.stopped: return []
        result = self._request('/api/state')
        state = result.get('state', {}) if result.get('ok') else {}
        # Preserve only the server's player-facing event/message text.
        messages = state.get('messages') or state.get('recentMessages') or []
        return [GameEvent(str(m), 'game') for m in messages if isinstance(m, str)]
    def read_state(self):
        result = self._request('/api/state')
        state = result.get('state', {}) if result.get('ok') else {}
        if self.requirements is not None: state = {**state, 'requirements': self.requirements}
        return state
    def propose(self, action): pass
    def send_approved(self, action):
        if self.stopped: return SendResult(False, 'transport stopped')
        try:
            result = self._request('/api/command', 'POST', {'command': action.command})
            if result.get('ok') is not True: return SendResult(False, str(result.get('error', 'command rejected')))
            raw = tuple(m for m in result.get('messages', []) if isinstance(m, dict))
            self.last_messages = raw
            for msg in raw:
                if isinstance(msg.get('requirements'), list): self.requirements = msg['requirements']
            messages = tuple(str(m.get('msg', '')) for m in raw if m.get('msg'))
            return SendResult(True, 'command accepted', messages)
        except Exception as error:
            self.stop(); return SendResult(False, f'{type(error).__name__}: {error}')
    def stop(self): self.stopped = True

def replay(lines: list[str]):
    for i, line in enumerate(lines, 1): yield parse_transcript(line, observation_id=i)
