from __future__ import annotations
import time
from dataclasses import dataclass
from .action_space import Action, allowed_action
from .observation import Observation

@dataclass(frozen=True)
class Proposal:
    action: Action
    reason: str
    state: str
    confidence: float

def propose(obs: Observation, *, combat_command: str = "attack", ttl: float = 5) -> Proposal:
    now = time.time()
    if obs.status != "ready" or obs.safe_mode is not True: return Proposal(allowed_action("wait", expires_at=now+ttl), "uncertain or unsafe state", "STOP", 1.0)
    if obs.in_combat is True: return Proposal(allowed_action("combat", combat_command=combat_command, expires_at=now+ttl), "combat target observed", "COMBAT", .65)
    return Proposal(allowed_action("look", expires_at=now+ttl), "no actionable combat state", "OBSERVE", .55)
