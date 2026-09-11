"""Hermes owns authentication, incoming Signal messages, and reply delivery."""
import asyncio
import json
from urllib.request import Request, urlopen
from urllib.error import URLError

BASE = "http://127.0.0.1:8765"

def request_fly(raw_args):
    command = raw_args.strip()
    if command in ("help", ""):
        return "NotFlyGuy: /fly status, pause, resume, reset, or <goal>. Local view: " + BASE
    try:
        if command != "status":
            payload = json.dumps({"message": command}).encode()
            with urlopen(Request(BASE + "/api/signal", data=payload,
                                 headers={"Content-Type": "application/json"}), timeout=5) as response:
                json.load(response)
            if command.lower() in ("pause", "resume", "reset"):
                return f"NotFlyGuy queued simulation command: {command.lower()}\n{BASE}"
            return f"NotFlyGuy queued your goal: {command}\n{BASE}\nMotor learning is still in development."
        with urlopen(BASE + "/api/state", timeout=5) as response:
            state = json.load(response)
        return (f"NotFlyGuy · {'paused' if state.get('paused') else 'running'} · step {state['timestep']} · goal: {state['goal']}\n"
                f"FlyBody image available: {bool(state.get('frame'))}\n{BASE}")
    except (URLError, TimeoutError, ValueError, KeyError):
        return "NotFlyGuy's local habitat is unavailable. Start the habitat server and retry /fly status."

async def handle(raw_args):
    return await asyncio.to_thread(request_fly, raw_args)

def register(ctx):
    ctx.register_command("fly", handler=handle, description="Watch and command NotFlyGuy",
                         args_hint="status | <goal>")
