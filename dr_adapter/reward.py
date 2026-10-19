"""Experiment scoring; this does not change DragonRealms progression rules."""
from __future__ import annotations

from dataclasses import dataclass
import math

REWARD_VERSION = 'balanced-requirements-exp/3'

def useful_exp_reward(before, after, target=2):
    raw = before.get('requirements')
    if not isinstance(raw, dict) or raw.get('circle') != target: return 0.0
    total = 0.0
    for skill, previous in before.get('skills', {}).items():
        current = after.get('skills', {}).get(skill, {})
        values = [previous.get('absorbedExp'), current.get('absorbedExp'), previous.get('rank'), previous.get('exp'), previous.get('nextRankExp')]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values): continue
        old, new, rank, partial, rank_cost = values
        if rank_cost <= 0 or new <= old: continue
        eligible = [row for row in raw.get('rows', []) if skill in row.get('eligible', [])
                    and isinstance(row.get('have'), (int, float)) and isinstance(row.get('need'), (int, float))
                    and row['have'] < row['need'] and rank < row['need']]
        if not eligible: continue
        weight = max(1 - row['have'] / row['need'] for row in eligible)
        # Count absorption once, cap credit at the next rank boundary and at
        # 0.1 per transition. Pool drain itself gets no second reward.
        total += .1 * weight * min(new - old, max(0, rank_cost - partial)) / rank_cost
    return min(.1, total)

def requirement_rows(state, target_circle=2):
    raw = state.get('requirements')
    if isinstance(raw, dict):
        if raw.get('circle', target_circle) != target_circle: return None
        raw = raw.get('rows')
    if not isinstance(raw, list): return None
    rows = {}
    for index, row in enumerate(raw):
        if not isinstance(row, dict): return None
        have, need = row.get('have'), row.get('need')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (have, need)): return None
        if have < 0 or need <= 0: return None
        key = row.get('label', str(index))
        if key in rows: return None
        rows[key] = (have, need)
    return rows

@dataclass(frozen=True)
class Score:
    reward: float
    gap: int | None
    reached_target: bool
    reason: str
    components: dict | None = None

def requirement_gap(state: dict, target_circle: int = 2) -> int | None:
    """Count unmet authoritative requirement rows returned by the DR API."""
    rows = requirement_rows(state, target_circle)
    return None if rows is None else sum(have < need for have, need in rows.values())

def score_transition(before: dict | None, after: dict, *, command_ok=True,
                     unsafe=False, target_circle=2, kill=False) -> Score:
    if unsafe: return Score(-10.0, requirement_gap(after, target_circle), False, 'unsafe-stop', {'safety': -10.0})
    p = after.get('player', {})
    reached = p.get('circle', 0) >= target_circle
    if p.get('hp', 1) <= 0: return Score(-10.0, requirement_gap(after, target_circle), False, 'death', {'death': -10.0})
    if reached:
        reward = 100.0 if (before or {}).get('player', {}).get('circle', 0) < target_circle else 0.0
        return Score(reward, 0, True, 'target-circle', {'target_circle': reward})
    old_gap = requirement_gap(before or {}, target_circle)
    new_gap = requirement_gap(after, target_circle)
    components = {'failure': -1.0} if not command_ok else {}
    old_rows = requirement_rows(before or {}, target_circle)
    new_rows = requirement_rows(after, target_circle)
    comparable = (old_rows is not None and new_rows is not None and old_rows.keys() == new_rows.keys()
                  and all(old_rows[k][1] == new_rows[k][1] for k in old_rows))
    if comparable and old_rows:
        components['useful_exp'] = useful_exp_reward(before or {}, after, target_circle)
        progress = excess = 0.0
        for key, (have, need) in new_rows.items():
            previous = old_rows[key][0]
            # Squared normalized deficits give lagging requirements more weight.
            old_deficit = max(0.0, 1 - previous / need)
            new_deficit = max(0.0, 1 - have / need)
            progress += old_deficit ** 2 - new_deficit ** 2
            excess += max(0.0, (have - need) / need) - max(0.0, (previous - need) / need)
        components['balanced_progress'] = 10.0 * progress / len(old_rows)
        components['requirement_closure'] = float(old_gap - new_gap)
        # Only penalize extra ranks when no useful requirement progress occurred.
        components['overtraining'] = -min(.25, max(0.0, excess) / len(old_rows)) if progress <= 0 else 0.0
        if kill: components['useful_kill'] = .1 if progress > 0 else 0.0
    # Unknown requirements earn no invented progress or blanket kill/step bonus.
    shaped = sum(components.values())
    return Score(shaped, new_gap, False, 'progress' if shaped > 0 else 'command-failed' if not command_ok else 'no-progress', components)
