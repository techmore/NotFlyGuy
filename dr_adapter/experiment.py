"""Bounded experiment against the real DR server with an isolated game DB."""
import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from .fly_bridge import FlyBridge
from .pilot import FlyPilot
from .reward import REWARD_VERSION
from .transport import DragonRealmsHTTPTransport


def copy_checkpoint(source, destination):
    """Snapshot explicit learning artifacts without modifying the source."""
    files = ('connectome.npz', 'policy.json')
    if not all((source / name).is_file() for name in files):
        raise ValueError('Checkpoint requires connectome.npz and policy.json')
    hashes = {}
    for name in files:
        output = destination / name
        shutil.copyfile(source / name, output)
        hashes[name] = hashlib.sha256(output.read_bytes()).hexdigest()
    return hashes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--minutes', type=float, default=20)
    parser.add_argument('--mode', choices=('train', 'frozen'), default='train')
    parser.add_argument('--checkpoint-root', type=Path, help='Explicit saved connectome.npz and policy.json to resume/evaluate')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.private.mkdir(parents=True, exist_ok=False)
    manifest = {'run_id': args.output.name, 'status': 'starting', 'pid': os.getpid(),
                'started_at': time.time(), 'minutes': args.minutes, 'target_circle': 2,
                'reward_version': REWARD_VERSION, 'mode': args.mode, 'race': 'gortog', 'guild': 'barbarian',
                'boost': 1, 'world': f'http://127.0.0.1:{args.port}',
                'comparison': 'development experiment; no baseline claim',
                'code_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    def save():
        manifest['updated_at'] = time.time()
        tmp = args.output / 'manifest.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(args.output / 'manifest.json')
    save()
    server = None
    transport = None
    def stop(*_):
        if transport: transport.stop()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if args.checkpoint_root:
            manifest['checkpoint_hashes'] = copy_checkpoint(args.checkpoint_root, args.private)
            manifest['initialization'] = 'saved-checkpoint'
        else:
            manifest['initialization'] = 'fresh-seed-zero'
        save()
        env = {**os.environ, 'PORT': str(args.port), 'DR_ENABLE_API': '1',
               'DR_DB_PATH': str((args.private / 'game.db').resolve())}
        with (args.private / 'server.log').open('w') as log:
            server = subprocess.Popen(['node', 'server/index.js'], cwd=args.game_root,
                                      env=env, stdout=log, stderr=log)
        manifest['server_pid'] = server.pid
        save()
        for _ in range(40):
            if server.poll() is not None: raise RuntimeError('Experiment server exited at startup')
            try:
                with urlopen(manifest['world'] + '/', timeout=1) as response:
                    if response.status == 200: break
            except OSError: time.sleep(.25)
        else: raise RuntimeError('Experiment server did not become ready')
        token = secrets.token_hex(8)
        transport = DragonRealmsHTTPTransport.bootstrap(manifest['world'], 'fly' + token,
                                                        secrets.token_urlsafe(24), 'FlyTrial')
        manifest['status'] = 'running'
        save()
        bridge = FlyBridge(checkpoint=args.private / 'connectome.npz')
        pilot = FlyPilot(transport, bridge, args.output / 'transitions.jsonl', args.minutes, 2,
                         args.private / 'policy.json', mode=args.mode)
        original_write = pilot.write
        def write(row):
            row = dict(row)
            row.pop('messages', None)
            original_write(row)
            manifest['step'] = pilot.step
            save()
        pilot.write = write
        pilot.run()
        manifest['status'] = pilot.ledger.data['last']['status']
    except KeyboardInterrupt:
        manifest['status'] = 'stopped'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if transport: transport.stop()
        if server and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        manifest['finished_at'] = time.time()
        save()

if __name__ == '__main__': main()
