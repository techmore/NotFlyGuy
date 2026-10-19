"""Bounded autonomous DR pilot using the real HTTP transport."""
from __future__ import annotations
import argparse, json, time
import re
from pathlib import Path
from .action_space import allowed_action
from .fly_bridge import FlyBridge
from .observation import from_game_state
from .transport import DragonRealmsHTTPTransport
from .learning import LinearActionLearner
from .training import TrainingLedger
from .curriculum import field_candidates, equipment_proposal, purchase_proposal
from .navigation import ObservedMap
from .reward import requirement_gap
from .outcome import settle_outcome

TOWN = {'square': [], 'bazaar': ['w','w'], 'bank_plaza': ['w','w','w','w'], 'hall_barbarian': ['s','s','w','e']}
# Observed-route path from square through the dense Crossing rooms into the
# sewer entrance. Keep this explicit and validate each move against API state.
TO_HUNT = ['nw','nw','nw','n','n','n','n','d']
TO_SQUARE = ['up','e','e','se']

class FlyPilot:
    def __init__(self, transport, bridge, log_path, minutes=10, target=2, learner_path=None, mode='train'):
        if mode not in ('train', 'frozen'): raise ValueError('mode must be train or frozen')
        self.mode = mode
        self.t, self.bridge, self.log = transport, bridge, Path(log_path); self.deadline=time.time()+minutes*60; self.target=target; self.step=0; self.last_room=None; self.failures=0; self.route=[]; self.last_messages=(); self.wait_until=0; self.last_action_at=0; self.kills=0; self.errand=None; self.train_idx=0; self.market_idx=0; self.bought_gear=False
        self.learner = LinearActionLearner(learner_path or self.log.with_name('fly-policy.json'))
        self.connectome_updates = 0
        self.navigation = ObservedMap()
        self.market_stock = []
        self.ledger = TrainingLedger(self.log.with_name('training-state.json'))
        self.ledger.start_episode(target=target, mode=mode)
        self.log.parent.mkdir(parents=True, exist_ok=True)
    def write(self, row):
        from .reward import REWARD_VERSION
        row['reward_version'] = REWARD_VERSION
        with self.log.open('a') as f: f.write(json.dumps(row)+'\n')
    def learn_transition(self, command, features, reward):
        if self.mode == 'frozen':
            return ({'enabled': False, 'updates': self.learner.updates},
                    {'enabled': False, 'updates': self.bridge.updates, 'weight_delta_rms': 0.0})
        return self.learner.update(command, features, reward), self.bridge.learn(reward)
    def choose(self, state, obs):
        p=state.get('player',{}); room=state.get('room') or {}; combat=state.get('combat'); rid=room.get('id','unknown')
        self.navigation.observe(room)
        if p.get('circle', 0) >= self.target: return 'wait', 'target-reached'
        if not rid or not p: return 'wait', 'unknown-state'
        health = p.get('hp', 0) / max(1, p.get('maxHp', 1))
        if health < .35: self.recovering = True
        if getattr(self, 'recovering', False):
            if health < .85: return ('retreat' if combat else 'rest'), 'low-health'
            self.recovering = False
        if combat and combat.get('enemies'): return 'attack ' + str(combat['enemies'][0].get('name','')), 'combat'
        if p.get('guild') == 'barbarian' and p.get('abilities') == []:
            if rid == 'hall_barbarian': return 'learn dragon', 'requirement-ability'
            direction = self.navigation.next_step(rid, 'hall_barbarian', explore=True)
            return (direction, 'route-to-guild') if direction else ('wait', 'unknown-state')
        for corpse in state.get('skinnable', []):
            target = corpse.get('id')
            if isinstance(target, str) and target and all(c.isascii() and (c.isalnum() or c == '_') for c in target):
                return 'skin ' + target, 'loot'
        equipment = equipment_proposal(state, self.target)
        if equipment: return equipment, 'requirement-equipment'
        if requirement_gap(state, self.target) == 0:
            self.route.clear()
            if rid == 'hall_barbarian': return 'circle', 'circle-attempt'
            direction = self.navigation.next_step(rid, 'hall_barbarian', explore=True)
            return (direction, 'route-to-guild') if direction else ('wait', 'unknown-state')
        if rid == 'square' and not self.bought_gear and not self.route: self.route=['e','e','e','e','e','e']; self.errand='market'
        if rid == 'bazaar' and not self.route:
            self.market_stock = state.get('shopStock', [])
            purchase = purchase_proposal(state)
            if purchase: return purchase, 'starter-equipment'
            # The fresh-character purse is intentionally small.  Buy a
            # usable weapon and torso armor in an order the live shop can
            # actually fund; failed purchases must not be treated as progress.
            if 'shopStock' not in state: return 'wait', 'unknown-state'
            self.bought_gear=True
            # Gear purchases happen at the Bazaar.  Return through the same
            # six reversible rooms used by the outbound market route before
            # entering the hunt path from the Town Green.
            self.route=['w'] * 6 + list(TO_HUNT)
            return 'look', 'return-to-hunt'
        if self.errand == 'hall':
            if rid == 'hall_barbarian': self.errand = None
            else:
                direction = self.navigation.next_step(rid, 'hall_barbarian', explore=True)
                return (direction, 'route-to-guild') if direction else ('wait', 'unknown-state')
        if self.route: return self.route.pop(0), 'town-route'
        if rid == 'hall_barbarian':
            # Requirement-aware training pass. The game rejects already
            # satisfied rows; the next state/reward records that outcome.
            skills=('expertise','melee_mastery','parry','tactics','inner_fire')
            if self.train_idx < len(skills):
                skill=skills[self.train_idx]; self.train_idx += 1
                return 'train ' + skill, 'requirement-training'
            return 'circle', 'circle-attempt'
        if self.bought_gear and not self.route and purchase_proposal({**state, 'shopStock': self.market_stock}):
            direction = self.navigation.next_step(rid, 'bazaar')
            if direction: return direction, 'route-to-market'
        if rid == 'square' and not self.route: self.route=list(TO_HUNT)
        if self.route:
            return self.route.pop(0), 'route-to-hunt'
        if rid == 'nw_road': return 'n', 'route-to-hunt'
        if rid == 'temple_row': return 'd', 'route-to-hunt'
        visible = re.sub(r'\x1b\[[0-9;]*m', '', ' '.join(self.last_messages))
        creature = re.search(r'\bA\s+([a-z][a-z ]+?)\s+is here\b', visible, re.I)
        if rid.startswith('sewers') and creature: return 'attack ' + creature.group(1).strip(), 'combat'
        if rid.startswith('sewers') and (' is here' in visible or 'are here' in visible): return 'hunt', 'seek-target'
        if rid in ('sewers_1','sewers_2','sewers_3'): return ('n' if rid != 'sewers_3' else 's'), 'patrol'
        if rid == 'temple': return 's', 'respawn'
        return 'look', 'observe'
    def run(self):
        while time.time() < self.deadline and not self.t.stopped:
            try:
                if time.time() < self.wait_until: time.sleep(.1); continue
                state=self.t.read_state(); messages=list(getattr(self.t, 'last_messages', ()))
                player = state.get('player', {})
                if player.get('hp', 1) <= 0:
                    self.t.stop(); self.ledger.finish('death'); break
                if player.get('circle', 0) >= self.target:
                    self.write({'step': self.step, 'at': time.time(), 'circle': player['circle'], 'terminal': 'target'})
                    self.t.stop(); self.ledger.finish('target'); break
                self.last_messages=tuple((str(m.get('msg','')) if isinstance(m,dict) else str(m)) for m in messages if m)
                combat = state.get('combat') if isinstance(state, dict) else None
                timer = player.get('roundtimeSeconds', 0)
                if isinstance(timer, (int, float)) and timer > 0:
                    self.wait_until = time.time() + min(timer, 30)
                    continue
                obs=from_game_state(state, messages=messages, observation_id=self.step+1)
                fly=self.bridge.observe(obs)
                command, reason=self.choose(state,obs)
                # Learning selects only between the planner's validated action
                # and an inert wait; it can never invent command text.
                candidates = [command] if reason.startswith(('combat', 'town-route', 'route-', 'starter-', 'requirement-', 'circle-', 'loot', 'low-health', 'respawn', 'patrol', 'seek-')) else [command, 'wait']
                if reason in ('patrol', 'observe', 'seek-target'):
                    candidates = list(dict.fromkeys([command, *field_candidates(state, self.target)]))
                if reason == 'combat' and combat and player.get('guild') == 'barbarian':
                    if 'dragon' in player.get('abilities', []) and player.get('innerFire', 0) >= 20 and combat.get('dragonTicks', 0) <= 0:
                        candidates.append('form dragon')
                    rows = (state.get('requirements') or {}).get('rows', [])
                    if any(r.get('label') in ('expertise', 'tactics') and r.get('have', 0) < r.get('need', 0) for r in rows):
                        if time.time() >= getattr(self, 'analyze_after', 0): candidates.append('analyze flame')
                command, explored = self.learner.choose(candidates, fly['features'], explore=self.mode == 'train' and reason not in ('unknown-state', 'target-reached'))
                self.write({'step': self.step, 'decision_candidates': candidates, 'selected': command})
                kind = 'combat' if command.startswith('attack ') else 'move' if command in {'n','ne','e','se','s','sw','w','nw','u','d','up','down'} else 'equip' if command.startswith(('wield ','wear ')) else command.split()[0]
                if kind == 'move' and command not in ((state.get('room') or {}).get('exits') or {}):
                    self.write({'step': self.step, 'stop': 'unobserved-exit', 'command': command})
                    self.t.stop(); self.ledger.finish('unobserved-exit'); break
                action=allowed_action(kind, combat_command=command, movement_command=command, configured_command=command, expires_at=time.time()+2)
                if action is None or reason == 'unknown-state': self.t.stop(); self.write({'step':self.step,'stop':reason}); break
                result=self.t.send_approved(action); self.last_messages=result.messages
                self.last_action_at=time.time()
                if command.startswith('analyze '): self.analyze_after = self.last_action_at + 10
                # Use the live server timer below.  No fixed combat delay is
                # assumed: weapon, stance, and target can all change it.
                if command.startswith('attack '):
                    # The API prompt can arrive before the combat manager's
                    # playerTimer is visible on the next state read.  Keep a
                    # bounded weapon cooldown as a floor; the live RT/wait
                    # parser below may extend it when the server reports more.
                    self.wait_until=max(self.wait_until, self.last_action_at + .5)
                # Re-read after the command so prompt requirements and EXP
                # changes are included in the next reward snapshot.
                after_state = self.t.read_state()
                text=' '.join(result.messages).lower()
                rt=re.search(r'RT:\s*(\d+)', ' '.join(result.messages), re.I)
                wait=re.search(r'(?:wait|roundtime)\D+(\d+)\s+second', text, re.I)
                if rt: self.wait_until=max(self.wait_until, time.time()+float(rt.group(1))+.6)
                elif wait: self.wait_until=max(self.wait_until, time.time()+float(wait.group(1))+.6)
                # Retain this action's circuit features until its recovery
                # window ends so delayed rank changes inform this transition.
                after_state = settle_outcome(self.t, after_state, until=self.wait_until,
                                             deadline=self.deadline, target=self.target)
                # Combat ending can also mean fleeing or dying. It is not
                # sufficient evidence of a kill, so do not invent a kill bonus.
                kill = False
                if 'invalid' in text or 'cannot' in text or 'not here' in text: self.failures += 1
                from .reward import score_transition, REWARD_VERSION
                failed_text = ('cannot', 'do not have', 'invalid', 'not here', 'must be in', 'unable', 'must wait', 'skin what')
                command_ok = result.ok and not any(token in text for token in failed_text)
                score = score_transition(state, after_state, command_ok=command_ok, unsafe=self.t.stopped, target_circle=self.target, kill=kill)
                learning, connectome_learning = self.learn_transition(command, fly['features'], score.reward)
                self.connectome_updates = connectome_learning['updates']
                self.ledger.data.setdefault('initial_experience', state.get('experience'))
                self.ledger.data['experience'] = after_state.get('experience')
                self.ledger.data.setdefault('initial_skills', state.get('skills', {}))
                self.ledger.data['skills'] = after_state.get('skills', {})
                self.ledger.data['requirements'] = after_state.get('requirements')
                self.ledger.data['circle'] = after_state.get('player', {}).get('circle')
                self.ledger.data['reward_version'] = REWARD_VERSION
                self.ledger.transition(score.reward, components=score.components, updates=int(self.mode == 'train'))
                self.write({'step':self.step,'at':time.time(),'room':obs.location,'circle':state.get('player',{}).get('circle'),'hp':obs.health_fraction,'command':command,'reason':reason,'explored':explored,'fly':fly,'reward':score.reward,'reward_components':score.components,'requirement_gap':score.gap,'ok':result.ok,'outcome':result.message,'messages':list(result.messages),'learning':learning,'connectome_learning':connectome_learning})
                self.step += 1
                if score.reached_target:
                    self.write({'step': self.step, 'at': time.time(), 'circle': after_state.get('player', {}).get('circle'), 'terminal': 'target'})
                    self.t.stop(); self.ledger.finish('target'); break
                if not result.ok: self.failures += 1
                if self.failures >= 5: self.t.stop(); self.ledger.finish('failure-limit'); break
                time.sleep(.25)
            except Exception as e:
                self.write({'step':self.step,'error':type(e).__name__+': '+str(e)}); self.t.stop(); self.ledger.finish('exception'); break
        if self.ledger.data.get('last', {}).get('status') == 'running':
            self.ledger.finish('stopped' if self.t.stopped else 'timeout')
        self.t.stop()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dr-url',required=True); ap.add_argument('--dr-token'); ap.add_argument('--user'); ap.add_argument('--password'); ap.add_argument('--character',default='FlyPilot'); ap.add_argument('--circuit-root'); ap.add_argument('--minutes',type=float,default=10); ap.add_argument('--target',type=int,default=2); ap.add_argument('--log',default='data/dr-runs/fly.jsonl'); ap.add_argument('--learner', help='JSON checkpoint for the trainable action readout'); a=ap.parse_args()
    if a.dr_token: transport=DragonRealmsHTTPTransport(a.dr_url,a.dr_token)
    elif a.user and a.password: transport=DragonRealmsHTTPTransport.bootstrap(a.dr_url,a.user,a.password,a.character)
    else: ap.error('provide --dr-token or --user and --password')
    FlyPilot(transport, FlyBridge(a.circuit_root, checkpoint=Path(a.log).with_name('dr-connectome.npz')), a.log, a.minutes, a.target, a.learner).run()
if __name__ == '__main__': main()
