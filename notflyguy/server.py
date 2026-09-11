"""Dependency-free live dashboard server for the first NotFlyGuy slice."""
from __future__ import annotations
import json, math, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).parent / "web"

class State:
    def __init__(self):
        self.t = 0
        self.message = "waiting for Signal/Hermes"
        self.goal = "idle"
    def snapshot(self):
        self.t += 1
        return {"timestep": self.t, "x": .5 + .25*math.sin(self.t/18), "y": .5 + .2*math.cos(self.t/23), "heading": math.sin(self.t/18), "activity": .25 + .2*math.sin(self.t/9), "goal": self.goal, "last_message": self.message}
    def command(self, message):
        self.message, self.goal = message, message[:80] or "idle"

STATE = State()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def send_json(self, value):
        data = json.dumps(value).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/api/state": self.send_json(STATE.snapshot()); return
        if self.path == "/events":
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(STATE.snapshot())}\n\n".encode()); self.wfile.flush(); time.sleep(.1)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        path = WEB / ("index.html" if self.path == "/" else self.path.lstrip("/"))
        if path.is_file() and WEB in path.parents:
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(path.read_bytes()); return
        self.send_error(404)
    def do_POST(self):
        if self.path != "/api/signal": self.send_error(404); return
        length = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(length) or b"{}")
        message = str(body.get("message", "")).strip(); STATE.command(message); self.send_json({"ok": True, "message": message})

def main():
    print("NotFlyGuy live at http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()

if __name__ == "__main__": main()
