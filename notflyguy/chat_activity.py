"""Read-only, text-free activity projection from the local Hermes gateway log."""
import os
import re
import time
from pathlib import Path


class ChatActivity:
    def __init__(self):
        self.path = Path(os.environ.get('NOTFLYGUY_GATEWAY_LOG',
            str(Path.home() / '.hermes/profiles/not-rocky/logs/agent.log')))
        self.offset = self.path.stat().st_size if self.path.exists() else 0
        self.session = None
        self.state = {'phase': 'waiting', 'label': 'Waiting for a new Signal message', 'at': time.time()}

    def consume(self, line):
        phase = label = None
        if 'conversation turn:' in line and 'platform=signal' in line:
            match = re.search(r'session=([^ ]+)', line)
            self.session = match.group(1) if match else None
            phase, label = 'typing', 'Not-Rocky is preparing a reply'
        elif self.session and f'[{self.session}]' in line:
            if 'tool notflyguy_habitat completed' in line:
                phase, label = 'typing', 'Checked the live fly · preparing a reply'
            elif 'Tool skill_view returned error' in line and 'notflyguy_habitat' in line:
                phase, label = 'error', 'Tool mistaken for a skill · reply may be incorrect'
            elif 'Turn ended:' in line:
                phase, label = 'ready', 'Reply generated · awaiting Signal handoff'
        elif 'response ready: platform=signal' in line:
            phase, label = 'ready', 'Reply generated · awaiting Signal handoff'
        elif 'gateway.platforms.signal: [Signal] Sending response chunk' in line:
            phase, label = 'handoff', 'Sending to Signal · delivery not confirmed'
        if phase:
            self.state = {'phase': phase, 'label': label, 'at': time.time()}

    def snapshot(self):
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.offset = 0
            with self.path.open() as stream:
                stream.seek(self.offset)
                for line in stream:
                    self.consume(line)
                self.offset = stream.tell()
        except OSError:
            return {'phase': 'unavailable', 'label': 'Gateway activity unavailable'}
        if self.state['phase'] == 'typing' and time.time() - self.state['at'] > 120:
            return {'phase': 'unknown', 'label': 'No recent gateway update · response status unknown'}
        return dict(self.state)
