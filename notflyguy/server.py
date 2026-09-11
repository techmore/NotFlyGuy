"""Dependency-free live dashboard server for the first NotFlyGuy slice."""
from __future__ import annotations
import base64, io, json, time
from queue import SimpleQueue, Empty
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
        self.commands = SimpleQueue()
        self.paused = False
        self.env = walk_imitation() if walk_imitation else None
        self.env.reset() if self.env else None
        self.current = {"timestep": 0, "goal": self.goal, "last_message": self.message, "frame": None, "paused": False}
    def snapshot(self):
        return self.current

    def tick(self):
        while True:
            try:
                message = self.commands.get_nowait()
            except Empty:
                break
            self.message = message
            if message.lower() == "pause":
                self.paused = True
            elif message.lower() == "resume":
                self.paused = False
            elif message.lower() == "reset":
                if self.env:
                    self.env.reset()
                self.t = 0
            else:
                self.goal = message[:80] or "idle"
        if not self.paused:
            self.t += 1
        frame = self.current.get("frame")
        if self.env and np and Image:
            if not self.paused:
                self.env.step(np.zeros(self.env.action_spec().shape, dtype=np.float32))
            image = Image.fromarray(self.env.physics.render(camera_id=1, height=240, width=480))
            buf = io.BytesIO(); image.save(buf, format="JPEG", quality=70)
            frame = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        self.current = {"timestep": self.t, "goal": self.goal, "last_message": self.message, "frame": frame, "paused": self.paused, "simulation_seconds": float(self.env.physics.data.time) if self.env else None}
    def command(self, message):
        self.commands.put(message)

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
        path = (WEB / ("index.html" if self.path == "/" else self.path.lstrip("/"))).resolve()
        if path.is_file() and WEB in path.parents:
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(path.read_bytes()); return
        self.send_error(404)
    def do_POST(self):
        if self.path != "/api/signal": self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                self.send_error(413); return
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get("message"), str):
                self.send_error(400); return
            message = body["message"].strip()
            if not message:
                self.send_error(400); return
        except (ValueError, UnicodeError):
            self.send_error(400); return
        STATE.command(message); self.send_json({"ok": True, "queued": True, "message": message})

def main():
    import threading
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("NotFlyGuy live at http://127.0.0.1:8765")
    while True:
        STATE.tick()
        time.sleep(.1)

if __name__ == "__main__": main()
