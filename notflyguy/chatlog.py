"""Read-only loader for Hermes transcripts as structured group-chat events.

Hermes persists every conversation turn in a SQLite ``messages`` table joined
to a ``sessions`` table that carries ``source`` and ``chat_type``. This module
turns those rows into :class:`ChatEvent` records the rest of the project can
train on.

Privacy posture: this reads local files only, never opens a network
connection, and never writes message text into a report or artifact. Sender and
group identifiers are pseudonymised with a truncated SHA-256 digest so that a
trained policy and any exported statistics refer to stable opaque handles
rather than phone numbers or UUIDs. Pseudonymisation is *not* encryption and is
not a security boundary -- treat the database itself as the sensitive artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

# Pseudonymisation salt. Stable by default so handles are reproducible across
# runs; override with NOTFLYGUY_ID_SALT when sharing derived data.
DEFAULT_SALT = 'notflyguy/v1'
MAX_TEXT = 8000
AGENT_SENDER = 'agent'


@dataclass(frozen=True)
class ChatEvent:
    """One message in a conversation, with opaque participant identity."""

    session_id: str
    group_id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: float
    is_agent: bool
    addressed_to_agent: bool
    index: int
    chat_type: str = ''
    source: str = ''

    def to_public(self) -> dict:
        """Serialisable view with the message text deliberately omitted."""
        row = asdict(self)
        row.pop('text', None)
        row['text_length'] = len(self.text)
        return row


def pseudonym(value, salt: str = DEFAULT_SALT, length: int = 12) -> str:
    if value is None or str(value) == '':
        return 'unknown'
    digest = hashlib.sha256((salt + '\x00' + str(value)).encode('utf-8')).hexdigest()
    return digest[:length]


def default_db_paths() -> list[Path]:
    """Hermes databases present on this machine, most specific last."""
    override = os.environ.get('NOTFLYGUY_HERMES_DB')
    if override:
        return [Path(p).expanduser() for p in override.split(':') if p.strip()]
    home = Path.home() / '.hermes'
    paths: list[Path] = []
    shared = home / 'state.db'
    if shared.exists():
        paths.append(shared)
    profiles = home / 'profiles'
    if profiles.is_dir():
        paths.extend(sorted(p / 'state.db' for p in profiles.iterdir() if (p / 'state.db').exists()))
    return paths


def _connect(path) -> sqlite3.Connection:
    uri = f'file:{Path(path).expanduser().resolve()}?mode=ro'
    return sqlite3.connect(uri, uri=True)


def _plain_text(content) -> str:
    """Text of a message content (str or list of text parts); '' otherwise."""
    if isinstance(content, list):
        parts = [p.get('text', '') for p in content
                 if isinstance(p, dict) and p.get('type') == 'text']
        return '\n'.join(t for t in parts if t)
    return content if isinstance(content, str) else ''


def _origin(origin_json) -> dict:
    try:
        parsed = json.loads(origin_json) if origin_json else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _metadata(display_metadata) -> dict:
    return _origin(display_metadata)


def _addressed(text: str, aliases) -> bool:
    lowered = text.lower()
    return any(alias and alias.lower() in lowered for alias in aliases)


def load_events(path, chat_types=('group',), include_agent=True,
                aliases=('notflyguy', 'not-rocky', 'not rocky'), salt=DEFAULT_SALT,
                limit_sessions=None) -> list[ChatEvent]:
    """Load one database into a chronological list of :class:`ChatEvent`.

    ``chat_types`` filters on ``sessions.chat_type``. Pass ``None`` to accept
    every session regardless of type (useful for the 1:1 CLI history).
    """
    events: list[ChatEvent] = []
    connection = _connect(path)
    try:
        connection.row_factory = sqlite3.Row
        query = ('SELECT id, source, chat_type, user_id, display_name, origin_json '
                 'FROM sessions')
        params: list = []
        if chat_types is not None:
            placeholders = ','.join('?' for _ in chat_types)
            query += f' WHERE chat_type IN ({placeholders})'
            params.extend(chat_types)
        query += ' ORDER BY started_at'
        sessions = connection.execute(query, params).fetchall()
        if limit_sessions is not None:
            sessions = sessions[-limit_sessions:]
        for session in sessions:
            origin = _origin(session['origin_json'])
            owner = session['user_id'] or origin.get('user_id')
            owner_name = origin.get('user_name') or ''
            agent_name = session['display_name'] or 'NotFlyGuy'
            group_id = pseudonym(origin.get('chat_id') or session['id'], salt)
            rows = connection.execute(
                'SELECT role, content, timestamp, display_metadata, tool_calls '
                'FROM messages WHERE session_id = ? AND COALESCE(active, 1) = 1 '
                'AND compacted = 0 ORDER BY timestamp, id',
                (session['id'],),
            ).fetchall()
            index = 0
            for row in rows:
                role = row['role']
                if role not in ('user', 'assistant'):
                    continue
                if role == 'assistant' and not include_agent:
                    continue
                text = _plain_text(row['content']).strip()[:MAX_TEXT]
                if not text:
                    # Tool-only and empty assistant turns are not conversational.
                    continue
                is_agent = role == 'assistant'
                meta = _metadata(row['display_metadata'])
                sender_raw = AGENT_SENDER if is_agent else (meta.get('gateway_input_owner') or owner)
                if is_agent:
                    sender_name = agent_name
                elif owner and str(sender_raw) == str(owner):
                    sender_name = owner_name or 'owner'
                else:
                    sender_name = f'participant-{pseudonym(sender_raw, salt, 6)}'
                events.append(ChatEvent(
                    session_id=session['id'],
                    group_id=group_id,
                    sender_id=AGENT_SENDER if is_agent else pseudonym(sender_raw, salt),
                    sender_name=sender_name,
                    text=text,
                    timestamp=float(row['timestamp'] or 0.0),
                    is_agent=is_agent,
                    addressed_to_agent=_addressed(text, aliases),
                    index=index,
                    chat_type=session['chat_type'] or '',
                    source=session['source'] or '',
                ))
                index += 1
    finally:
        connection.close()
    return events


def load_all(paths=None, **kwargs) -> list[ChatEvent]:
    events: list[ChatEvent] = []
    for path in (paths if paths is not None else default_db_paths()):
        try:
            events.extend(load_events(path, **kwargs))
        except sqlite3.Error:
            continue
    events.sort(key=lambda event: (event.session_id, event.timestamp, event.index))
    return events


def corpus_stats(events) -> dict:
    """Aggregate shape of a corpus. Contains no message text."""
    sessions = {event.session_id for event in events}
    groups = {event.group_id for event in events}
    participants = {event.sender_id for event in events if not event.is_agent}
    user_messages = [event for event in events if not event.is_agent]
    agent_messages = [event for event in events if event.is_agent]
    addressed = [event for event in user_messages if event.addressed_to_agent]
    return {
        'events': len(events),
        'sessions': len(sessions),
        'groups': len(groups),
        'participants': len(participants),
        'user_messages': len(user_messages),
        'agent_messages': len(agent_messages),
        'addressed_to_agent': len(addressed),
        'mean_user_text_length': (sum(len(e.text) for e in user_messages) / len(user_messages)
                                  if user_messages else 0.0),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the Hermes group-chat corpus.')
    parser.add_argument('--db', action='append', help='database path (repeatable)')
    parser.add_argument('--chat-types', default='group',
                        help="comma-separated chat_type filter, or 'all'")
    args = parser.parse_args()
    paths = [Path(p) for p in args.db] if args.db else None
    chat_types = None if args.chat_types == 'all' else tuple(
        t for t in args.chat_types.split(',') if t)
    events = load_all(paths, chat_types=chat_types)
    stats = corpus_stats(events)
    stats['databases'] = [str(p) for p in (paths or default_db_paths())]
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
