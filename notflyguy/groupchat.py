"""Turn a real transcript into connectome-driven group-chat decision points.

Each user message is one decision: should the agent speak now, and how long
should the reply be? The fly's neural state is advanced causally -- only events
that happened before the decision are visible to it -- so nothing downstream
can leak the label.

:class:`ChatStateTracker` owns that causal state and is shared by the offline
trainer and the live dashboard, so a policy is trained on exactly the features
it will later be served.

The connectome contributes a fixed recurrent dynamical system that integrates
the message stream. The *drive* channels and the *labels* are modeling choices,
not measured biology; they are named here so that claim stays checkable.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .chatlog import ChatEvent
from .connectome import DRIVE_NAMES, ConnectomeState

# Normalisation constant for reply length, in characters.
REPLY_LENGTH_SCALE = 2000.0
SILENT_REPLY_LENGTH = -1.0

CONTEXT_NAMES = (
    'log_text_length',
    'is_question',
    'addressed_to_agent',
    'sender_is_new',
    'log_sender_prior_messages',
    'log_gap_seconds',
    'messages_since_agent_reply',
    'log_position',
    'distinct_senders_so_far',
    'uppercase_ratio',
    'log_mean_text_length',
    'agent_reply_rate_so_far',
    'is_group',
    'participant_count',
)


@dataclass(frozen=True)
class DecisionPoint:
    session_id: str
    group_id: str
    sender_id: str
    timestamp: float
    index: int
    context: tuple[float, ...]
    connectome: tuple[float, ...]
    respond: bool
    reply_length: float
    addressed: bool
    is_group: bool

    def to_public(self) -> dict:
        row = asdict(self)
        row['context'] = list(row['context'])
        row['connectome'] = list(row['connectome'])
        return row


def _question(text: str) -> float:
    return 1.0 if '?' in text else 0.0


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _normalise_length(length: int) -> float:
    return float(min(1.0, math.log1p(max(length, 0)) / math.log1p(REPLY_LENGTH_SCALE)))


def drive_for_user(text: str, addressed: bool, is_new_sender: bool, gap_seconds: float,
                   messages_since_reply: int) -> np.ndarray:
    """Modeled sensory injection for an incoming human message."""
    salience = (0.6 * min(1.0, math.log1p(len(text)) / math.log1p(600))
                + 0.6 * _question(text) + 0.6 * float(addressed))
    novelty = 1.0 if is_new_sender else 0.0
    recency = 1.0 / (1.0 + max(gap_seconds, 0.0) / 60.0)
    pressure = min(2.0, messages_since_reply / 4.0) + (0.5 if addressed else 0.0)
    return np.array([salience, novelty, recency, pressure, 0.0], dtype=float)


def drive_for_agent(text: str) -> np.ndarray:
    """Modeled injection for the agent's own turn (closure / relief)."""
    return np.array([0.0, 0.0, 0.0, -1.0, min(2.0, 0.5 + _normalise_length(len(text)))],
                    dtype=float)


class ChatStateTracker:
    """Causal per-session state feeding the connectome.

    Used identically offline (dataset construction) and online (live policy
    inference) so training and serving cannot drift apart.
    """

    def __init__(self, brain: ConnectomeState, session_id: str = '', group_id: str = '',
                 chat_type: str = ''):
        if brain is None:
            raise ValueError('ChatStateTracker requires a ConnectomeState')
        self.brain = brain
        self.session_id = session_id
        self.group_id = group_id
        self.chat_type = chat_type
        self.reset()

    def reset(self) -> None:
        self.brain.reset()
        self.seen_senders: set[str] = set()
        self.sender_counts: dict[str, int] = {}
        self.messages_since_reply = 0
        self.agent_replies = 0
        self.user_seen = 0
        self.text_lengths: list[int] = []
        self.last_timestamp: float | None = None
        self.position = 0
        self.decisions: list[dict] = []

    def _gap(self, timestamp: float) -> float:
        if self.last_timestamp is None:
            return 0.0
        return max(0.0, timestamp - self.last_timestamp)

    def observe_agent(self, text: str, timestamp: float = 0.0) -> np.ndarray:
        self.last_timestamp = timestamp
        self.position += 1
        self.brain.step(drive_for_agent(text))
        self.messages_since_reply = 0
        self.agent_replies += 1
        return self.brain.activity

    def observe_user(self, event: ChatEvent) -> tuple[np.ndarray, np.ndarray]:
        """Ingest a human message and return ``(context, connectome_features)``."""
        gap = self._gap(event.timestamp)
        self.last_timestamp = event.timestamp
        is_new = event.sender_id not in self.seen_senders
        addressed = bool(event.addressed_to_agent)
        prior = self.sender_counts.get(event.sender_id, 0)
        history = self.text_lengths[-10:]
        context = np.array([
            math.log1p(len(event.text)),
            _question(event.text),
            float(addressed),
            float(is_new),
            math.log1p(prior),
            math.log1p(gap),
            float(min(self.messages_since_reply, 12)),
            # Deliberately a function of message count only. Normalising by the
            # session's total length would require knowing the future and would
            # differ between offline training and live serving.
            math.log1p(self.position),
            math.log1p(len(self.seen_senders)),
            _uppercase_ratio(event.text),
            math.log1p(sum(history) / len(history)) if history else 0.0,
            self.agent_replies / max(self.user_seen + 1, 1),
            1.0 if event.chat_type == 'group' else 0.0,
            float(len(self.seen_senders)),
        ], dtype=float)
        self.position += 1
        self.brain.step(drive_for_user(event.text, addressed, is_new, gap, self.messages_since_reply))
        connectome = self.brain.features()
        self.seen_senders.add(event.sender_id)
        self.sender_counts[event.sender_id] = prior + 1
        self.messages_since_reply += 1
        self.user_seen += 1
        self.text_lengths.append(len(event.text))
        return context, connectome


def build_decision_points(events, brain: ConnectomeState | None = None,
                          seed: int = 0) -> list[DecisionPoint]:
    """Convert chronological events into labelled decision points."""
    if brain is None:
        brain = ConnectomeState(seed=seed)
    ordered = sorted(events, key=lambda e: (e.session_id, e.timestamp, e.index))
    sessions: dict[str, list[ChatEvent]] = {}
    for event in ordered:
        sessions.setdefault(event.session_id, []).append(event)

    points: list[DecisionPoint] = []
    for session_id, rows in sessions.items():
        tracker = ChatStateTracker(brain, session_id=session_id)
        for position, event in enumerate(rows):
            if event.is_agent:
                tracker.observe_agent(event.text, event.timestamp)
                continue
            context, connectome = tracker.observe_user(event)
            next_event = rows[position + 1] if position + 1 < len(rows) else None
            replied = bool(next_event is not None and next_event.is_agent)
            reply_length = _normalise_length(len(next_event.text)) if replied else SILENT_REPLY_LENGTH
            points.append(DecisionPoint(
                session_id=session_id,
                group_id=event.group_id,
                sender_id=event.sender_id,
                timestamp=event.timestamp,
                index=event.index,
                context=tuple(float(v) for v in context),
                connectome=tuple(float(v) for v in connectome),
                respond=replied,
                reply_length=reply_length,
                addressed=bool(event.addressed_to_agent),
                is_group=event.chat_type == 'group',
            ))
    return points


def feature_matrix(points, include_connectome: bool = True):
    """Stack decision points into a design matrix and label vectors."""
    context = np.array([p.context for p in points], dtype=float)
    if not include_connectome:
        X = context
    else:
        connectome = np.array([p.connectome for p in points], dtype=float)
        X = np.hstack([context, connectome])
    y_respond = np.array([1.0 if p.respond else 0.0 for p in points], dtype=float)
    y_length = np.array([p.reply_length if p.respond else np.nan for p in points], dtype=float)
    return X, y_respond, y_length


def feature_names(include_connectome: bool = True) -> tuple[str, ...]:
    names = CONTEXT_NAMES
    if include_connectome:
        names = names + tuple(f'connectome_{n}' for n in ConnectomeState.feature_names())
    return names


def dataset_stats(points) -> dict:
    total = len(points)
    responded = sum(1 for p in points if p.respond)
    group_points = [p for p in points if p.is_group]
    addressed = sum(1 for p in points if p.addressed)
    sessions = {p.session_id for p in points}
    groups = {p.group_id for p in points}
    return {
        'decision_points': total,
        'sessions': len(sessions),
        'groups': len(groups),
        'group_decision_points': len(group_points),
        'responded': responded,
        'respond_rate': responded / total if total else 0.0,
        'majority_class_rate': max(responded, total - responded) / total if total else 0.0,
        'addressed': addressed,
        'drive_channels': list(DRIVE_NAMES),
    }
