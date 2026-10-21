from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .observation import from_game_state, parse_transcript
from .policy import Proposal, propose
from .transport import GameTransport, MockTransport


def origin_ok(headers) -> bool:
    """Same-origin check for browser writes; non-browser clients send no Origin.

    ``/api/approve`` sends a real game command, so a page from any other origin
    open in the same browser must not be able to drive it. Mirrors
    ``notflyguy.server._origin_ok``.
    """
    origin = headers.get('Origin')
    if origin is None:
        return True
    host = headers.get('Host')
    if not host:
        return False
    parsed = urlsplit(origin)
    return parsed.scheme in ('http', 'https') and parsed.netloc == host

class Adapter:
    def __init__(self, transport: GameTransport, combat_command="attack", fly_bridge=None):
        self.transport, self.combat_command, self.fly_bridge = transport, combat_command, fly_bridge
        self.lock = threading.RLock(); self.observation = parse_transcript("", observation_id=0)
        self.proposal: Proposal | None = None; self.paused = False; self.armed = True
        self.last_event = "none"; self.audit: list[dict] = []; self.fly = None; self.last_reward = None
    def observe(self, text: str):
        with self.lock:
            self.observation = parse_transcript(text, observation_id=self.observation.observation_id + 1)
            self.last_event = self.observation.recent_events[-1] if self.observation.recent_events else "screen"
            fly_features = self.fly_bridge.observe(self.observation) if self.fly_bridge else None
            self.fly = fly_features
            if self.observation.status in ("dead", "disconnected", "login", "unknown"): self.stop()
            elif not self.paused and self.armed:
                self.proposal = propose(self.observation, combat_command=self.combat_command); self.transport.propose(self.proposal.action)
            snapshot = self.snapshot()
            return snapshot
    def observe_state(self, state: dict, messages=()):
        """Ingest the real structured API state without lossy screen parsing."""
        with self.lock:
            self.observation = from_game_state(state, messages=messages,
                                                observation_id=self.observation.observation_id + 1)
            self.last_event = self.observation.recent_events[-1] if self.observation.recent_events else 'state'
            fly_features = self.fly_bridge.observe(self.observation) if self.fly_bridge else None
            self.fly = fly_features
            if self.observation.status in ('dead', 'disconnected', 'login', 'unknown'):
                self.stop()
            elif not self.paused and self.armed:
                self.proposal = propose(self.observation, combat_command=self.combat_command)
                self.transport.propose(self.proposal.action)
            snapshot = self.snapshot()
            return snapshot
    def approve(self):
        with self.lock:
            p = self.proposal
            if self.paused or not self.armed or p is None or time.time() > p.action.expires_at: return {"ok": False, "error": "no live proposal"}
            if self.observation.status != "ready": return {"ok": False, "error": "state is not ready"}
            result = self.transport.send_approved(p.action); self.audit.append({"observation_id": self.observation.observation_id, "action": p.action.command, "approved": result.ok, "outcome": result.message})
            self.proposal = None; return {"ok": result.ok, "message": result.message}
    def stop(self):
        with self.lock: self.armed = False; self.proposal = None; self.transport.stop()
    def pause(self, paused=True):
        with self.lock:
            self.paused = paused
            self.proposal = None
            return self.snapshot()
    def snapshot(self):
        return {"observation": self.observation.as_dict(), "proposal": None if self.proposal is None else {"action": self.proposal.action.command, "state": self.proposal.state, "reason": self.proposal.reason, "confidence": self.proposal.confidence, "expires_at": self.proposal.action.expires_at}, "paused": self.paused, "armed": self.armed, "observer_only": not self.armed, "last_event": self.last_event, "fly": self.fly, "reward": self.last_reward, "audit": self.audit[-20:]}

class Handler(BaseHTTPRequestHandler):
    adapter: Adapter | None = None
    training_state = None
    def log_message(self, *_): pass
    def send_json(self, value, status=200):
        body=json.dumps(value).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/api/state": return self.send_json(self.adapter.snapshot())
        if self.path == "/api/training":
            try:
                if not self.training_state: return self.send_json({})
                path = Path(self.training_state)
                data = json.loads(path.read_text())
                data['updated_at'] = path.stat().st_mtime
                logs = list(path.parent.glob('*.jsonl'))
                if logs:
                    latest = max(logs, key=lambda item: item.stat().st_mtime)
                    with latest.open('rb') as stream:
                        size = latest.stat().st_size
                        stream.seek(max(0, size - 131072))
                        lines = stream.read().splitlines()
                    for line in reversed(lines):
                        try: row = json.loads(line)
                        except (ValueError, UnicodeDecodeError): continue
                        if 'step' not in row or 'at' not in row: continue
                        data['latest_transition'] = {key: row.get(key) for key in (
                            'step', 'at', 'room', 'circle', 'hp', 'reason',
                            'reward', 'requirement_gap', 'connectome_learning')}
                        data['latest_log'] = latest.stem
                        break
                return self.send_json(data)
            except (OSError, ValueError): return self.send_json({})
        if self.path == "/":
            body = Path(__file__).with_name('dashboard.html').read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header('Cache-Control', 'no-store'); self.send_header("Content-Length", str(len(body))); self.end_headers(); return self.wfile.write(body)
        return self.send_json({"error":"not found"},404)
    def do_POST(self):
        if not origin_ok(self.headers): return self.send_json({"error": "origin not allowed"}, 403)
        try: length=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(min(length,16384)))
        except (ValueError, json.JSONDecodeError): return self.send_json({"error":"invalid JSON"},400)
        if self.path == "/api/observe" and isinstance(body,dict) and isinstance(body.get("text"),str): return self.send_json(self.adapter.observe(body["text"]))
        if self.path == "/api/propose": return self.send_json(self.adapter.snapshot())
        if self.path == "/api/approve": return self.send_json(self.adapter.approve())
        if self.path == "/api/pause": return self.send_json(self.adapter.pause(body.get('paused', True)))
        if self.path == "/api/stop": self.adapter.stop(); return self.send_json(self.adapter.snapshot())
        return self.send_json({"error":"not found"},404)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=8787); ap.add_argument("--combat-command",default="attack"); ap.add_argument('--dr-url', default=None); ap.add_argument('--dr-token', default=None); ap.add_argument('--user', default=None); ap.add_argument('--password', default=None); ap.add_argument('--character', default='FlyDashboard'); ap.add_argument('--circuit-root', default=None); ap.add_argument('--training-state', default=None); args=ap.parse_args()
    if args.dr_token and (args.user or args.password): ap.error('use --dr-token or --user/--password, not both')
    if args.user or args.password:
        if not (args.user and args.password and args.dr_url): ap.error('--dr-url, --user, and --password are required together')
    transport = MockTransport()
    bridge = None
    if args.dr_url:
        from .fly_bridge import FlyBridge
        from .transport import DragonRealmsHTTPTransport
        transport = (DragonRealmsHTTPTransport(args.dr_url, args.dr_token)
                     if args.dr_token else
                     DragonRealmsHTTPTransport.bootstrap(args.dr_url, args.user, args.password, args.character))
        bridge = FlyBridge(Path(args.circuit_root) if args.circuit_root else None)
    adapter=Adapter(transport, args.combat_command, bridge); Handler.adapter=adapter
    Handler.training_state = str(Path(args.training_state)) if args.training_state else None
    if args.dr_url:
        def poll():
            while not transport.stopped:
                try:
                    adapter.observe_state(transport.read_state(), getattr(transport, 'last_messages', ()))
                except Exception:
                    adapter.stop(); break
                time.sleep(.5)
        threading.Thread(target=poll, daemon=True).start()
    print(f"DR adapter observer dashboard: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()
DASHBOARD = '''<!doctype html><meta name="viewport" content="width=device-width"><title>Fly training</title>
<style>body{font:15px system-ui;background:#10141b;color:#e6edf3;max-width:1000px;margin:2rem auto;padding:0 1rem}section{background:#18212b;border:1px solid #304052;border-radius:10px;padding:1rem;margin:1rem 0}h1{margin-bottom:.2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem}.card{background:#202c39;padding:.7rem;border-radius:7px}small{color:#9fb0c1}pre{white-space:pre-wrap;max-height:260px;overflow:auto}button{padding:.6rem 1rem;background:#b42318;color:white;border:0;border-radius:6px}</style>
<h1>NotFlyGuy × DragonRealms</h1><small id="mode">connecting…</small><section><h2>Game state</h2><div class="grid" id="game"></div></section><section><h2>Fly circuit</h2><div class="grid" id="fly"></div></section><section><h2>Current proposal</h2><div id="proposal"></div><button onclick="stop()">Emergency stop</button></section><section><h2>Recent command audit</h2><pre id="audit"></pre></section>
<script>async function state(){try{let s=await fetch('/api/state').then(r=>r.json()),o=s.observation||{},p=s.proposal||{},f=s.fly||{};let put=(id,x)=>document.getElementById(id).innerHTML=x;let card=(k,v)=>`<div class=card><small>${k}</small><br><b>${v??'unknown'}</b></div>`;put('mode',s.armed?'AUTONOMOUS PILOT ARMED':'STOPPED / DISARMED');put('game',[card('Location',o.location),card('Circle',o.circle||'?'),card('Health',o.health_fraction==null?'unknown':Math.round(o.health_fraction*100)+'%'),card('Combat',o.in_combat),card('RT',o.roundtime_seconds),card('Last event',s.last_event)].join(''));put('fly',[card('Namespace',f.namespace),card('Activity RMS',f.activity_rms?.toFixed?.(5)),card('Features',f.features?.length),card('Reward',s.reward)].join(''));put('proposal',p.action?`${card('Command',p.action)}${card('Reason',p.reason)}${card('Confidence',Math.round(p.confidence*100)+'%')}`:'No current proposal');put('audit',(s.audit||[]).slice().reverse().map(x=>JSON.stringify(x)).join('\\n')||'No commands yet')}catch(e){document.getElementById('mode').textContent='connection error: '+e}}async function stop(){await fetch('/api/stop',{method:'POST',headers:{'content-type':'application/json'},body:'{}'});state()}state();setInterval(state,500)</script>
<script>async function training(){try{let t=await fetch('/api/training').then(r=>r.json()),l=t.last||{},card=(k,v)=>`<div class=card><small>${k}</small><br><b>${v??'unknown'}</b></div>`;document.body.insertAdjacentHTML('afterbegin',`<section><h2>Training ledger</h2><div class=grid>${['Episodes|'+(t.episodes??0),'Transitions|'+(t.transitions??0),'Updates|'+(t.updates??0),'Cumulative reward|'+(t.total_reward??0).toFixed(2),'Last episode|'+(l.episode??'—')+' / '+(l.status??'—'),'Reward components|'+JSON.stringify(t.reward_components||{})].map(x=>{let [k,v]=x.split('|');return card(k,v)}).join('')}</div></section>`)}catch(e){}}training();setInterval(training,2000)</script>'''

if __name__ == "__main__": main()
