"""Dependency-free live dashboard server for the first NotFlyGuy slice."""
from __future__ import annotations
import base64, io, json, math, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).parent / "web"

try:
    import numpy as np
    from PIL import Image
    from flybody.fly_envs import walk_imitation
except ImportError:
    np = Image = walk_imitation = None

class State:
    def __init__(self):
        self.t = 0
        self.message = "waiting for Signal/Hermes"
        self.goal = "idle"
        self.env = walk_imitation() if walk_imitation else None
        self.env.reset() if self.env else None
        self.current = {"timestep": 0, "x": .5, "y": .5, "heading": 0, "activity": .25, "goal": self.goal, "last_message": self.message, "frame": None}
    def snapshot(self):
        return self.current

    def tick(self):
        self.t += 1
        frame = None
        if self.env and np and Image:
            self.env.step(np.zeros(59, dtype=np.float32))
            image = Image.fromarray(self.env.physics.render(camera_id=1, height=240, width=480))
            buf = io.BytesIO(); image.save(buf, format="JPEG", quality=70)
            frame = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        self.current = {"timestep": self.t, "x": .5 + .25*math.sin(self.t/18), "y": .5 + .2*math.cos(self.t/23), "heading": math.sin(self.t/18), "activity": .25 + .2*math.sin(self.t/9), "goal": self.goal, "last_message": self.message, "frame": frame}
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
    import threading
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("NotFlyGuy live at http://127.0.0.1:8765")
    while True:
        STATE.tick()
        time.sleep(.1)

if __name__ == "__main__": main()
