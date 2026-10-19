from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import time
from typing import Any

SCHEMA = "notflyguy.dr.observation/1"

@dataclass(frozen=True)
class Observation:
    timestamp: float
    location: str = "unknown"
    in_combat: bool | None = None
    target_present: bool | None = None
    health_fraction: float | None = None
    fatigue_fraction: float | None = None
    stamina_fraction: float | None = None
    mana_fraction: float | None = None
    roundtime_seconds: float | None = None
    balance: str = "unknown"
    wounds: str = "unknown"
    status: str = "unknown"
    recent_events: list[str] = field(default_factory=list)
    safe_mode: bool = True
    observation_id: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **asdict(self)}

def _fraction(value: str, maximum: str) -> float | None:
    try:
        n, d = float(value), float(maximum)
        return n / d if d > 0 and 0 <= n <= d else None
    except ValueError:
        return None

def parse_transcript(text: str, *, timestamp: float | None = None, observation_id: int = 0) -> Observation:
    """Parse only explicit facts. Missing or malformed fields remain unknown."""
    if not isinstance(text, str):
        text = ""
    events: list[str] = []
    low = text.lower()
    patterns = {
        "death": r"\b(dead|you have died|death)\b",
        "disconnect": r"\b(disconnected|connection lost)\b",
        "login_prompt": r"(login|password):?\s*$",
        "error": r"\b(error|invalid command|not understood)\b",
        "combat": r"\b(attacking|engaged|combat|foe|creature)\b",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, low, re.MULTILINE): events.append(name)
    room = "unknown"
    match = re.search(r"(?:You are in|Location:|Room:)\s*([^\n.]+)", text, re.I)
    if match: room = match.group(1).strip()
    vals: dict[str, float | None] = {}
    for key, labels in (("health_fraction", "health|hp"), ("fatigue_fraction", "fatigue"),
                        ("stamina_fraction", "stamina"), ("mana_fraction", "mana")):
        m = re.search(rf"(?:{labels})\s*[: ]\s*(\d+)\s*/\s*(\d+)", text, re.I)
        vals[key] = _fraction(m.group(1), m.group(2)) if m else None
    rt = re.search(r"(?:roundtime|RT)\s*[: ]\s*(\d+(?:\.\d+)?)", text, re.I)
    status = "dead" if "death" in events else "disconnected" if "disconnect" in events else "login" if "login_prompt" in events else "unknown"
    if not any(e in events for e in ("death", "disconnect", "login_prompt", "error")) and (room != "unknown" or any(v is not None for v in vals.values()) or "combat" in events): status = "ready"
    in_combat = True if "combat" in events else False if status in ("dead", "disconnected", "login") else None
    return Observation(timestamp=time.time() if timestamp is None else timestamp, location=room,
        in_combat=in_combat, target_present=True if "combat" in events else None,
        roundtime_seconds=float(rt.group(1)) if rt else None, status=status,
        recent_events=events[-12:], safe_mode=True, observation_id=observation_id, **vals)

def from_game_state(state: dict, *, messages=(), timestamp=None, observation_id=0) -> Observation:
    """Build an observation from DR's authenticated structured /api/state result."""
    p = state.get('player') if isinstance(state, dict) else None
    room = state.get('room') if isinstance(state, dict) else None
    combat = state.get('combat') if isinstance(state, dict) else None
    if not isinstance(p, dict): return parse_transcript('', timestamp=timestamp, observation_id=observation_id)
    def frac(a, b):
        try: return float(a) / float(b) if float(b) > 0 else None
        except (TypeError, ValueError): return None
    events = [str(m) for m in messages if isinstance(m, str)][-12:]
    low = ' '.join(events).lower()
    fatal = any(x in low for x in ('died', 'disconnected', 'login', 'error'))
    return Observation(timestamp=time.time() if timestamp is None else timestamp,
        location=str(room.get('name', room.get('id', 'unknown'))) if isinstance(room, dict) else 'unknown',
        in_combat=bool(combat) if combat is not None else None,
        target_present=bool(combat and combat.get('enemies')) if isinstance(combat, dict) else None,
        health_fraction=frac(p.get('hp'), p.get('maxHp')),
        stamina_fraction=None, mana_fraction=frac(p.get('mana'), p.get('maxMana')),
        wounds='known' if p.get('wounds') else 'none',
        status='dead' if 'died' in low else 'disconnected' if 'disconnect' in low else 'ready' if room else 'unknown',
        recent_events=events, safe_mode=not fatal, observation_id=observation_id)
