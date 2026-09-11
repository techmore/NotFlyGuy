"""Hermes owns authentication, incoming Signal messages, and reply delivery.

This plugin adds three things to a Hermes turn:

1. A connectome/policy reading for the incoming message, fetched from the local
   habitat, injected as *advisory* style guidance.
2. A dashboard feed so the NotFlyGuy GUI shows what the fly is doing.
3. The agent's own reply, fed back to the circuit so its state reflects closure.

Reply gating is deliberately not attempted. Hermes documents ``pre_llm_call``
as a directive hook whose valid return values are *joined and injected* into
the user message, and ``post_llm_call`` as an observer whose return is ignored.
There is no plugin surface that suppresses a send, so the policy can only
advise. Do not describe it as deciding whether the fly replies.
"""
import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE = os.environ.get('NOTFLYGUY_BASE', 'http://127.0.0.1:8765')
IDENTITY = 'NotFlyGuy'
ALIASES = ('notflyguy', 'not-rocky', 'not rocky')

PERSONA = (
    'For this conversation, speak as NotFlyGuy, the little fly at a keyboard. '
    'Be a friendly, curious, lightly mischievous conversational companion, not a diagnostic endpoint. '
    'Respond to what the person actually said and follow the conversation. '
    'For greetings or "how is it going", give a short natural social reply, perhaps a small fly joke; '
    'do not report running state, steps, neurons, reward, controller details, or viewer URLs. '
    'Do not call habitat tools or load skills for small talk. Do not recycle a fixed catchphrase. '
    'Do not introduce yourself as Rocky or Not-Rocky; that is the transport account, not your character. '
    'Technical details belong only in answers to explicit technical questions. '
    'Do not pretend the biological connectome creates your language or that animated typing is learned behavior. '
    'Do not claim actions, perception, memories, or training outcomes you do not have evidence for. '
    'Earlier telemetry-heavy greetings were a configuration mistake, not the desired conversation style.'
)

TECHNICAL = re.compile(r'\b(neurons?|connectome|telemetry|simulation|controller|reward|rms|viewer|habitat|status)\b', re.I)


def _request(path, payload=None, timeout=2.0, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(BASE + path, data=data, method=method,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def publish_chat(phase, text):
    """Feed the local dashboard. Never raises; the dashboard is not critical."""
    try:
        _request('/api/chat-events', {'phase': phase, 'text': str(text)[:6000]})
    except (OSError, ValueError, URLError):
        pass


def _db_paths():
    override = os.environ.get('NOTFLYGUY_HERMES_DB')
    if override:
        return [Path(p).expanduser() for p in override.split(':') if p.strip()]
    home = Path.home() / '.hermes'
    paths = [home / 'state.db']
    profiles = home / 'profiles'
    if profiles.is_dir():
        paths.extend(sorted(p / 'state.db' for p in profiles.iterdir()))
    return [p for p in paths if p.exists()]


def session_chat_type(session_id):
    """Look up whether this session is a group chat. '' when unknown."""
    if not session_id:
        return ''
    for path in _db_paths():
        try:
            uri = f'file:{path.resolve()}?mode=ro'
            with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
                row = connection.execute(
                    'SELECT chat_type FROM sessions WHERE id = ?', (session_id,)).fetchone()
            if row and row[0]:
                return str(row[0])
        except sqlite3.Error:
            continue
    return ''


def observe_group_message(session_id, sender_id, text, chat_type=''):
    """Ask the habitat for a policy reading. Returns None when unavailable."""
    try:
        response = _request('/api/group-message', {
            'text': str(text)[:2000],
            'sender': sender_id or session_id or 'unknown',
            'sender_name': 'group member',
            'group': session_id or 'signal-group',
            'source': 'signal',
            'aliases': list(ALIASES),
            'chat_type': chat_type or 'group',
        })
        decision = response.get('decision') if isinstance(response, dict) else None
        return decision if isinstance(decision, dict) else None
    except (OSError, ValueError, URLError):
        return None


def observe_agent_reply(text):
    try:
        _request('/api/agent-reply', {'text': str(text)[:2000]})
    except (OSError, ValueError, URLError):
        pass


def style_guidance(decision, chat_type):
    """Turn a policy reading into advisory wording for the model."""
    if not decision or not decision.get('policy_loaded'):
        return ''
    probability = decision.get('respond_probability')
    length = decision.get('predicted_reply_length')
    if probability is None:
        return ''
    where = 'group chat' if (chat_type or 'group') == 'group' else 'conversation'
    lines = [
        f'The fly control policy read this {where} message with internal state '
        f'from the measured connectome (circuit activity RMS {decision.get("connectome", {}).get("rms", 0):.3f}).',
    ]
    if probability < 0.5:
        lines.append(
            f'Its learned readout would hold back here (p={probability:.2f}). This is advisory only: '
            'you must still answer, but prefer a light, brief contribution over a long reply.'
        )
    else:
        lines.append(f'Its learned readout would speak here (p={probability:.2f}).')
    if length is not None:
        if length > 0.7:
            lines.append('It expects a fuller reply; a few sentences are appropriate.')
        elif length < 0.4:
            lines.append('It expects a short reply; keep it to a sentence or two.')
    lines.append('Do not mention the policy numbers or the connectome unless explicitly asked.')
    return '\n'.join(lines)


def conversation_context(user_message, platform='', session_id='', sender_id='', chat_type=''):
    context = PERSONA
    decision = None
    if str(platform).lower() == 'signal':
        decision = observe_group_message(session_id, sender_id, user_message, chat_type)
        guidance = style_guidance(decision, chat_type)
        if guidance:
            context += '\n' + guidance
    if TECHNICAL.search(str(user_message)):
        context += ('\nFresh technical data for this explicit inquiry (use only relevant fields):\n'
                    + request_fly('status'))
    return context


def before_reply(platform='', user_message='', session_id='', sender_id='', **kwargs):
    if str(platform).lower() != 'signal':
        return
    publish_chat('typing', user_message)
    chat_type = session_chat_type(session_id)
    return {'context': conversation_context(user_message, platform=platform,
                                            session_id=session_id, sender_id=sender_id,
                                            chat_type=chat_type)}


def after_reply(platform='', assistant_response='', **kwargs):
    if str(platform).lower() != 'signal':
        return
    publish_chat('ready', assistant_response)
    observe_agent_reply(assistant_response)


def request_fly(raw_args):
    command = raw_args.strip()
    if command in ('help', ''):
        return 'NotFlyGuy: /fly status, pause, resume, reset, or <goal>. Local view: ' + BASE
    try:
        if command != 'status':
            payload = {'message': command}
            _request('/api/signal', payload, timeout=5.0)
            if command.lower() in ('pause', 'resume', 'reset'):
                return f'NotFlyGuy queued simulation command: {command.lower()}\n{BASE}'
            return (f'NotFlyGuy queued your goal: {command}\n{BASE}\n'
                    'Motor learning is still in development.')
        state = _request('/api/state', timeout=5.0)
        chat = state.get('chat') or {}
        lines = [
            f"NotFlyGuy · {'paused' if state.get('paused') else 'running'} · "
            f"step {state.get('timestep')} · goal: {state.get('goal')}",
            f"Controller: {state.get('controller', 'unavailable')}",
            f"Circuit: {state.get('neurons', 0)} neurons / {state.get('edges', 0)} edges",
            f"Activity RMS: {state.get('activity_rms')} · reward: {state.get('reward')}",
            f"Group-chat policy: {'loaded' if chat.get('policy_loaded') else 'not trained'}",
            f"Local Mac viewer: {BASE}",
        ]
        return '\n'.join(lines)
    except (URLError, TimeoutError, ValueError, KeyError, OSError):
        return 'NotFlyGuy\'s local habitat is unavailable. Start the habitat server and retry /fly status.'


async def handle(raw_args):
    return await asyncio.to_thread(request_fly, raw_args)


def habitat_tool(args, **kwargs):
    command = args.get('command', 'status')
    if command not in ('status', 'pause', 'resume', 'reset'):
        return json.dumps({'error': 'Unsupported habitat command'})
    try:
        return json.dumps({'response': request_fly(command)})
    except Exception:
        return json.dumps({'error': 'Habitat request failed'})


def register(ctx):
    ctx.register_hook('pre_llm_call', before_reply)
    ctx.register_hook('post_llm_call', after_reply)
    ctx.register_command('fly', handler=handle, description='Watch and command NotFlyGuy',
                         args_hint='status | <goal>')
    ctx.register_tool(
        name='notflyguy_habitat', toolset='notflyguy', handler=habitat_tool,
        schema={'name': 'notflyguy_habitat',
                'description': 'Read live NotFlyGuy simulation status, or pause, resume, or reset its FlyBody environment. The live controller uses a measured MaleCNS motor subgraph with simplified, untrained rate dynamics. It is not a full brain emulation or a language model; meaningful learned locomotion is not established.',
                'parameters': {'type': 'object', 'properties': {
                    'command': {'type': 'string', 'enum': ['status', 'pause', 'resume', 'reset']}},
                    'required': ['command'], 'additionalProperties': False}})
