"""Durable training-run ledger for the DR adapter."""
from __future__ import annotations
import json, time
from pathlib import Path

class TrainingLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {'schema': 'notflyguy.dr.training/1', 'episodes': 0,
                     'transitions': 0, 'updates': 0, 'total_reward': 0.0,
                     'reward_components': {}, 'last': None}
        if self.path.exists():
            try: self.data.update(json.loads(self.path.read_text()))
            except (OSError, ValueError): pass

    def start_episode(self, *, mode='train', target=2):
        self.data['episodes'] = int(self.data.get('episodes', 0)) + 1
        self.data['last'] = {'episode': self.data['episodes'], 'mode': mode,
                             'target': target, 'started_at': time.time(),
                             'steps': 0, 'reward': 0.0, 'status': 'running'}
        self.save()

    def transition(self, reward, *, components=None, updates=0):
        last = self.data['last'] or {}
        last['steps'] = int(last.get('steps', 0)) + 1
        last['reward'] = float(last.get('reward', 0.0)) + float(reward)
        self.data['transitions'] = int(self.data.get('transitions', 0)) + 1
        self.data['updates'] = int(self.data.get('updates', 0)) + int(updates)
        self.data['total_reward'] = float(self.data.get('total_reward', 0.0)) + float(reward)
        for key, value in (components or {}).items():
            self.data['reward_components'][key] = float(self.data['reward_components'].get(key, 0.0)) + float(value)
        self.save()

    def finish(self, status):
        if self.data['last']:
            self.data['last']['status'] = status
            self.data['last']['finished_at'] = time.time()
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))
