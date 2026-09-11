"""Headless habitat: simulator, connectome, chat policy, and observable state.

This module owns all mutable state and is deliberately free of HTTP so that it
can be driven directly from tests. The HTTP layer in :mod:`notflyguy.server` is
a thin adapter over :class:`Habitat`.

Threading contract: every mutation happens under ``self._lock``. ``snapshot()``
performs a single reference read of a dict that is rebuilt atomically at the
end of each tick, so HTTP threads never block the simulator and never observe a
half-updated frame.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from queue import Empty, SimpleQueue

from .chat_activity import ChatActivity
from .chatlog import ChatEvent, pseudonym, DEFAULT_SALT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIRCUIT_ROOT = ROOT / 'data' / 'malecns-v1.0'
DEFAULT_POLICY = ROOT / 'data' / 'training' / 'chat-policy-v1' / 'policy.npz'
DEFAULT_SHADOW_LOG = ROOT / 'data' / 'training' / 'chat-policy-v1' / 'shadow.jsonl'

FEED_LIMIT = 60
CONTROLLER_LABEL = 'MaleCNS motor subgraph · untrained rate model'


class Habitat:
    """Owns the simulated body, the measured circuit, and the chat policy."""

    def __init__(self, env=None, controller=None, chat_brain=None, policy=None,
                 chat_activity=None, policy_path=DEFAULT_POLICY,
                 shadow_log=DEFAULT_SHADOW_LOG, frame_size=(240, 480),
                 frame_quality=70, frame_interval=2):
        self._lock = threading.RLock()
        self.env = env
        self.controller = controller
        self.chat_brain = chat_brain
        self.policy = policy
        self.policy_path = Path(policy_path) if policy_path else None
        self.shadow_log = Path(shadow_log) if shadow_log else None
        self.chat_activity = chat_activity or ChatActivity()
        self.frame_size = frame_size
        self.frame_quality = frame_quality
        self.frame_interval = max(int(frame_interval), 1)

        self.t = 0
        self.message = 'waiting for Signal/Hermes'
        self.goal = 'idle'
        self.paused = False
        self.episode = 1
        self.reward = None
        self.simulation_seconds = None
        self.commands: SimpleQueue = SimpleQueue()
        self.chat_events: SimpleQueue = SimpleQueue()
        self.latest_chat = None
        self.frame_jpeg: bytes | None = None
        self.frame_id = 0
        self.sim_error: str | None = None
        self._ticks = 0

        # Live group-chat shadow state.
        self.chat_tracker = None
        self.chat_last_decision = None
        self.chat_feed: list[dict] = []
        self.chat_counts = {'groups': 0, 'messages': 0, 'decisions': 0}
        self._known_groups: set[str] = set()
        self._corpus: dict | None = None
        self._corpus_at = 0.0
        if chat_brain is not None and policy is not None:
            self._reset_chat_tracker()
        self._snapshot = self._build_snapshot()

    # -- construction ------------------------------------------------------
    @classmethod
    def create(cls, circuit_root=DEFAULT_CIRCUIT_ROOT, policy_path=DEFAULT_POLICY,
               shadow_log=DEFAULT_SHADOW_LOG, with_body=True, **kwargs) -> 'Habitat':
        """Best-effort construction: a missing body, circuit, or policy degrades."""
        env = controller = chat_brain = policy = None
        if with_body:
            try:
                from flybody.fly_envs import walk_on_ball
                env = walk_on_ball()
                env.reset()
            except Exception:
                env = None
        if env is not None:
            try:
                from .circuit import MotorCircuitController
                from .motor_projection import limits_from_spec
                names = env.physics.named.model.actuator_ctrlrange.axes.row.names
                controller = MotorCircuitController(
                    circuit_root, names, action_limits=limits_from_spec(env.action_spec()))
            except Exception:
                controller = None
        try:
            from .connectome import ConnectomeState
            chat_brain = ConnectomeState(circuit_root, seed=1)
        except Exception:
            chat_brain = None
        if chat_brain is not None and policy_path:
            try:
                from .chat_policy import ChatPolicy
                policy = ChatPolicy.load(policy_path)
            except Exception:
                policy = None
        return cls(env=env, controller=controller, chat_brain=chat_brain, policy=policy,
                   policy_path=policy_path, shadow_log=shadow_log, **kwargs)

    def _reset_chat_tracker(self):
        from .groupchat import ChatStateTracker
        self.chat_tracker = ChatStateTracker(self.chat_brain, session_id='live',
                                             group_id='live', chat_type='group')

    # -- simulation --------------------------------------------------------
    def command(self, message: str) -> dict:
        message = str(message).strip()
        self.commands.put(message)
        return {'queued': True, 'message': message}

    def publish_chat(self, phase: str, text: str) -> None:
        self.chat_events.put({'phase': phase, 'text': text})

    def _drain_queues(self) -> None:
        while True:
            try:
                event = self.chat_events.get_nowait()
            except Empty:
                break
            prefix = 'You' if event['phase'] == 'typing' else 'NotFlyGuy'
            self.latest_chat = {'phase': event['phase'],
                                'label': f"{prefix}: {event['text']}", 'at': time.time()}
        while True:
            try:
                message = self.commands.get_nowait()
            except Empty:
                break
            self.message = message
            lowered = message.lower()
            if lowered == 'pause':
                self.paused = True
            elif lowered == 'resume':
                self.paused = False
            elif lowered == 'reset':
                self.reset()
            else:
                self.goal = message[:80] or 'idle'

    def reset(self) -> None:
        if self.env is not None:
            self.env.reset()
        if self.controller is not None:
            self.controller.reset()
        self.t = 0
        self.reward = None

    def _render(self) -> None:
        if self.env is None:
            return
        try:
            import numpy as np
            from PIL import Image
            height, width = self.frame_size
            pixels = self.env.physics.render(camera_id=1, height=height, width=width)
            image = Image.fromarray(np.asarray(pixels, dtype='uint8'))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.frame_quality)
            self.frame_jpeg = buffer.getvalue()
            self.frame_id += 1
        except Exception:
            self.frame_jpeg = None

    def tick(self) -> dict:
        """Advance one control step and atomically publish a new snapshot."""
        with self._lock:
            self._ticks += 1
            self._drain_queues()
            if self.env is not None and not self.paused:
                try:
                    import numpy as np
                    ball = self.env.physics.named.data.qvel['ball']
                    action = (self.controller.step(ball) if self.controller is not None
                              else np.zeros(self.env.action_spec().shape))
                    timestep = self.env.step(action)
                    self.reward = float(timestep.reward)
                    self.simulation_seconds = float(self.env.physics.data.time)
                    self.t += 1
                    if timestep.last():
                        self.env.reset()
                        if self.controller is not None:
                            self.controller.reset()
                        self.episode += 1
                except Exception as error:  # surfaced, never silently swallowed
                    self.sim_error = f'{type(error).__name__}: {error}'
            if self._ticks % self.frame_interval == 0:
                self._render()
            self._snapshot = self._build_snapshot()
            return self._snapshot

    # -- live group chat ---------------------------------------------------
    def observe_group_message(self, payload: dict) -> dict:
        """Advance the chat circuit for one message and log the policy decision."""
        with self._lock:
            source = str(payload.get('source', 'simulator'))
            group_raw = str(payload.get('group') or 'default')
            group_id = pseudonym(group_raw, DEFAULT_SALT)
            if group_id not in self._known_groups:
                self._known_groups.add(group_id)
                self.chat_counts['groups'] = len(self._known_groups)
            sender_raw = str(payload.get('sender') or 'unknown')
            text = str(payload.get('text') or '')[:2000]
            aliases = payload.get('aliases') or ('notflyguy', 'not-rocky', 'not rocky')
            addressed = any(a.lower() in text.lower() for a in aliases if a)
            event = ChatEvent(
                session_id=group_id, group_id=group_id,
                sender_id=pseudonym(sender_raw, DEFAULT_SALT),
                sender_name=str(payload.get('sender_name') or 'participant')[:60],
                text=text, timestamp=float(payload.get('timestamp') or time.time()),
                is_agent=False, addressed_to_agent=addressed,
                index=int(self.chat_counts['messages']),
                chat_type=str(payload.get('chat_type') or 'group'), source=source,
            )
            decision = {'source': source, 'group_id': group_id,
                        'sender_name': event.sender_name, 'addressed': addressed,
                        'at': time.time(), 'policy_loaded': self.policy is not None}
            if self.chat_tracker is None:
                decision['error'] = 'connectome unavailable'
            else:
                import numpy as np
                from .groupchat import CONTEXT_NAMES
                context, connectome = self.chat_tracker.observe_user(event)
                features = np.concatenate([context, connectome])
                decision['connectome'] = {
                    'rms': float(np.sqrt(np.mean(self.chat_brain.activity ** 2))),
                    'peak': float(self.chat_brain.activity.max()),
                    'mean_motor': float(self.chat_brain.motor_activity.mean()),
                    'features': [round(float(v), 5) for v in connectome],
                }
                if self.policy is not None:
                    probability = float(self.policy.predict_proba(features.reshape(1, -1))[0])
                    length = float(self.policy.predict_length(features.reshape(1, -1))[0])
                    decision['respond_probability'] = round(probability, 4)
                    decision['predicted_reply_length'] = round(max(0.0, min(1.0, length)), 4)
                    decision['would_respond'] = bool(probability >= 0.5)
                decision['context'] = dict(zip(CONTEXT_NAMES, (round(float(v), 4) for v in context)))
            self.chat_counts['messages'] += 1
            self.chat_counts['decisions'] += 1
            self.chat_last_decision = decision
            self._append_feed(decision)
            self._append_shadow(decision, source)
            self._snapshot = self._build_snapshot()
            return decision

    def observe_agent_reply(self, text: str = '') -> None:
        with self._lock:
            if self.chat_tracker is not None:
                self.chat_tracker.observe_agent(str(text)[:2000], time.time())

    def _append_feed(self, decision: dict) -> None:
        self.chat_feed.append(decision)
        if len(self.chat_feed) > FEED_LIMIT:
            del self.chat_feed[:-FEED_LIMIT]

    def _append_shadow(self, decision: dict, source: str) -> None:
        """Persist predictions only -- never message text."""
        if self.shadow_log is None:
            return
        row = {key: decision.get(key) for key in
               ('at', 'source', 'group_id', 'addressed', 'respond_probability',
                'predicted_reply_length', 'would_respond', 'policy_loaded')}
        try:
            self.shadow_log.parent.mkdir(parents=True, exist_ok=True)
            with self.shadow_log.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        except OSError:
            pass

    # -- observation -------------------------------------------------------
    def _build_snapshot(self) -> dict:
        import numpy as np
        neurons = edges = 0
        activity_rms = None
        if self.controller is not None:
            neurons = len(self.controller.ids)
            edges = self.controller.recurrent.nnz
            activity_rms = float(np.sqrt(np.mean(self.controller.activity ** 2)))
        chat_rms = None
        if self.chat_brain is not None:
            chat_rms = float(np.sqrt(np.mean(self.chat_brain.activity ** 2)))
        gateway = self.latest_chat or self.chat_activity.snapshot()
        return {
            'timestep': self.t,
            'goal': self.goal,
            'last_message': self.message,
            'paused': self.paused,
            'simulation_seconds': self.simulation_seconds,
            'controller': CONTROLLER_LABEL if self.controller is not None else 'unavailable',
            'neurons': neurons,
            'edges': edges,
            'activity_rms': activity_rms,
            'reward': self.reward,
            'episode': self.episode,
            'frame_id': self.frame_id,
            'has_frame': self.frame_jpeg is not None,
            'sim_error': self.sim_error,
            'chat_activity': gateway,
            'chat': {
                'policy_loaded': self.policy is not None,
                'policy_path': str(self.policy_path) if self.policy_path else None,
                'circuit_loaded': self.chat_brain is not None,
                'chat_activity_rms': chat_rms,
                'counts': dict(self.chat_counts),
                'last': self.chat_last_decision,
                'feed': list(self.chat_feed[-20:]),
            },
        }

    def snapshot(self) -> dict:
        return self._snapshot

    def reports(self) -> dict:
        """Training reports on disk, keyed by run directory name."""
        output = {}
        training = ROOT / 'data' / 'training'
        if not training.is_dir():
            return output
        for report in sorted(training.glob('*/report.json')):
            try:
                output[report.parent.name] = json.loads(report.read_text())
            except (OSError, ValueError):
                continue
        return output

    def corpus_report(self, ttl: float = 60.0) -> dict:
        """Corpus readiness, cached: it reads the Hermes databases."""
        with self._lock:
            now = time.time()
            if self._corpus is not None and now - self._corpus_at < ttl:
                return self._corpus
            try:
                from .corpus import status as corpus_status
                self._corpus = corpus_status()
            except Exception as error:
                self._corpus = {'error': f'{type(error).__name__}: {error}'}
            self._corpus_at = now
            return self._corpus


def build_habitat(**kwargs) -> Habitat:
    """Construct a live habitat; used by the server entry point."""
    return Habitat.create(**kwargs)
