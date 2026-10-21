from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionName = Literal["wait", "look", "status", "retreat", "combat", "move", "hunt", "skin", "rest", "train", "circle", "buy", "equip", "forage", "perform", "study", "appraise", "analyze", "learn", "form"]
SAFE = frozenset(("wait", "look", "status", "retreat", "combat", "move", "hunt", "skin", "rest", "train", "circle", "buy", "equip", "forage", "perform", "study", "appraise"))
MOVES = frozenset(('n','ne','e','se','s','sw','w','nw','u','d','up','down'))

@dataclass(frozen=True)
class Action:
    name: ActionName
    command: str
    expires_at: float

def allowed_action(name: str, *, combat_command: str = "attack", movement_command: str | None = None, configured_command: str | None = None, expires_at: float = 0) -> Action | None:
    if name in ('learn', 'form'):
        if configured_command != name + ' dragon': return None
        return Action(name=name, command=configured_command, expires_at=expires_at)
    if name == 'analyze':
        if configured_command not in ('analyze flame', 'analyze accuracy', 'analyze damage'): return None
        return Action(name='analyze', command=configured_command, expires_at=expires_at)
    if name not in SAFE: return None
    if name in ('appraise', 'skin'):
        parts = (configured_command or '').split()
        if len(parts) != 2 or parts[0] != name or not all(c.isascii() and (c.isalnum() or c == '_') for c in parts[1]): return None
        return Action(name=name, command=' '.join(parts), expires_at=expires_at)
    command = combat_command if name == "combat" else movement_command if name == 'move' and movement_command else configured_command if name in ('train','circle','buy','equip') and configured_command else name
    if name == 'move' and command not in MOVES: return None
    if not isinstance(command, str) or not command or any(c in command for c in "\r\n;|&"):
        return None
    return Action(name=name, command=command, expires_at=expires_at)  # type: ignore[arg-type]
