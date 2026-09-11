"""Shared fixtures. No test here requires FlyBody, MuJoCo, or a live server."""
import sqlite3

import pytest


def _create_schema(connection):
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            chat_type TEXT,
            user_id TEXT,
            display_name TEXT,
            origin_json TEXT,
            started_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL,
            tool_calls TEXT,
            tool_name TEXT,
            display_metadata TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            compacted INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def _add_session(connection, session_id, source, chat_type, user_id, display_name, origin,
                 started_at):
    connection.execute(
        'INSERT INTO sessions (id, source, chat_type, user_id, display_name, origin_json, '
        'started_at) VALUES (?,?,?,?,?,?,?)',
        (session_id, source, chat_type, user_id, display_name, origin, started_at),
    )


def _add_message(connection, session_id, role, content, timestamp, owner=None,
                 active=1, compacted=0):
    metadata = None
    if owner is not None:
        import json
        metadata = json.dumps({'gateway_input_owner': owner})
    connection.execute(
        'INSERT INTO messages (session_id, role, content, timestamp, display_metadata, '
        'active, compacted) VALUES (?,?,?,?,?,?,?)',
        (session_id, role, content, timestamp, metadata, active, compacted),
    )


@pytest.fixture
def hermes_db(tmp_path):
    """A small synthetic Hermes database with two groups and one direct session.

    The first group deliberately contains a back-to-back user pair, so the
    'did the agent reply' label has both a positive and a negative example.
    """
    import json
    path = tmp_path / 'state.db'
    connection = sqlite3.connect(path)
    _create_schema(connection)
    _add_session(connection, 's-group-1', 'signal', 'group', '+15550001', 'not-rocky',
                 json.dumps({'platform': 'signal', 'chat_id': 'group:AAA', 'chat_type': 'group',
                             'user_id': '+15550001', 'user_name': 'Owner'}), 1000.0)
    _add_session(connection, 's-group-2', 'signal', 'group', '+15550002', 'not-rocky',
                 json.dumps({'platform': 'signal', 'chat_id': 'group:BBB', 'chat_type': 'group',
                             'user_id': '+15550002', 'user_name': 'Owner'}), 2000.0)
    _add_session(connection, 's-direct-1', 'cli', None, 'local', 'not-rocky',
                 json.dumps({'platform': 'cli'}), 3000.0)

    # Group 1: reply, reply, then two user messages with no reply between them.
    _add_message(connection, 's-group-1', 'user', 'notflyguy, you around?', 1001.0, '+15550001')
    _add_message(connection, 's-group-1', 'assistant', 'Always. Buzzing and ready.', 1002.0)
    _add_message(connection, 's-group-1', 'user', 'Pick a topic.', 1003.0, '+15550009')
    _add_message(connection, 's-group-1', 'assistant', 'How about flight paths?', 1004.0)
    _add_message(connection, 's-group-1', 'user', 'never mind', 1005.0, '+15550009')
    _add_message(connection, 's-group-1', 'user', 'actually, tell me later', 1006.0, '+15550001')
    _add_message(connection, 's-group-1', 'assistant', 'Noted.', 1007.0)
    # A tool-only assistant row and a compacted row must be ignored.
    _add_message(connection, 's-group-1', 'assistant', '', 1008.0)
    _add_message(connection, 's-group-1', 'assistant', 'compacted away', 1009.0, compacted=1)

    _add_message(connection, 's-group-2', 'user', 'status?', 2001.0, '+15550002')
    _add_message(connection, 's-group-2', 'assistant', 'Running fine.', 2002.0)

    _add_message(connection, 's-direct-1', 'user', 'hello there', 3001.0)
    _add_message(connection, 's-direct-1', 'assistant', 'Hello!', 3002.0)
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def group_events(hermes_db):
    from notflyguy.chatlog import load_events
    return load_events(hermes_db, chat_types=('group',))


@pytest.fixture
def decision_points(group_events):
    from notflyguy.groupchat import build_decision_points
    return build_decision_points(group_events, seed=0)


class FakeTimestep:
    def __init__(self, reward=0.0, last=False):
        self.reward = reward
        self._last = last

    def last(self):
        return self._last


class FakeEnv:
    """Just enough MuJoCo surface for Habitat.tick, with no simulator installed."""

    class _Vector:
        def __getitem__(self, key):
            import numpy as np
            return np.zeros(3)

    class _Data:
        def __init__(self):
            self.qvel = FakeEnv._Vector()

    class _Named:
        def __init__(self):
            self.data = FakeEnv._Data()

    class _Physics:
        def __init__(self):
            self.named = FakeEnv._Named()
            self.data = type('Data', (), {'time': 0.0})()

        def render(self, **kwargs):
            raise RuntimeError('no renderer in tests')

    def __init__(self, reward=0.25, last_after=None):
        self.physics = FakeEnv._Physics()
        self.steps = 0
        self.resets = 0
        self._reward = reward
        self._last_after = last_after

    def reset(self):
        self.resets += 1
        self.steps = 0

    def step(self, action):
        self.steps += 1
        last = self._last_after is not None and self.steps >= self._last_after
        return FakeTimestep(self._reward, last)

    def action_spec(self):
        import numpy as np
        return type('Spec', (), {'shape': (4,)})()


@pytest.fixture
def fake_env():
    return FakeEnv()
