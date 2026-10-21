import numpy as np

from dr_adapter.action_space import allowed_action
from dr_adapter.dashboard import Adapter
from dr_adapter.fly_bridge import FlyBridge
from dr_adapter.learning import LinearActionLearner
from dr_adapter.observation import from_game_state, parse_transcript
from dr_adapter.policy import propose
from dr_adapter.reward import requirement_gap, score_transition
from dr_adapter.transport import MockTransport


def test_parser_explicit_fields_and_schema():
    obs = parse_transcript("Location: Crossing\nHP: 80/100 Mana: 20/40\nYou are attacking a goblin. RT: 3")
    assert obs.location == "Crossing"
    assert obs.health_fraction == .8
    assert obs.mana_fraction == .5
    assert obs.roundtime_seconds == 3
    assert obs.in_combat is True
    assert obs.as_dict()["schema"] == "notflyguy.dr.observation/1"

def test_parser_never_guesses_missing_values():
    obs = parse_transcript("a confusing screen")
    assert obs.location == "unknown"
    assert obs.health_fraction is None
    assert obs.in_combat is None
    assert obs.status == "unknown"

def test_structured_api_state_is_authoritative():
    obs = from_game_state({'player': {'hp': 50, 'maxHp': 100, 'mana': 4, 'maxMana': 8, 'wounds': []},
                           'room': {'id': 'square', 'name': 'Town Green'},
                           'combat': {'enemies': [{'name': 'goblin'}]}}, messages=[])
    assert obs.location == 'Town Green'
    assert obs.health_fraction == .5
    assert obs.in_combat is True
    assert obs.target_present is True

def test_allowlist_rejects_injection_and_unknown():
    assert allowed_action("chat") is None
    assert allowed_action("combat", combat_command="attack;drop all") is None
    assert allowed_action("move", movement_command="nw").command == "nw"

def test_policy_stops_on_uncertainty_and_proposes_combat():
    uncertain = parse_transcript("")
    assert propose(uncertain).state == "STOP"
    combat = parse_transcript("Location: Arena\nHP: 10/10\nattacking orc")
    assert propose(combat).action.command == "attack"

def test_approval_expiry_and_emergency_stop():
    transport = MockTransport()
    adapter = Adapter(transport)
    adapter.observe("Location: Arena\nHP: 10/10")
    adapter.proposal = propose(adapter.observation, ttl=-1)
    assert adapter.approve()["ok"] is False
    adapter.observe("Location: Arena\nHP: 10/10")
    adapter.stop()
    assert adapter.approve()["ok"] is False
    assert not transport.sent

def test_death_disconnect_and_unknown_stop_transport():
    transport = MockTransport()
    adapter = Adapter(transport)
    adapter.observe("You have died.")
    assert transport.stopped
    assert adapter.snapshot()["proposal"] is None
    assert adapter.snapshot()["armed"] is False

def test_pause_blocks_approval_until_fresh_observation():
    transport = MockTransport()
    adapter = Adapter(transport)
    adapter.observe('Location: Arena\nHP: 10/10')
    adapter.pause()
    assert adapter.approve()['ok'] is False
    assert transport.sent == []
    adapter.pause(False)
    assert adapter.approve()['ok'] is False
    adapter.observe('Location: Arena\nHP: 10/10')
    assert adapter.approve()['ok'] is True

def test_reward_uses_requirement_gap_and_target_as_primary_signal():
    before = {'requirements': [{'have': 1, 'need': 2}, {'have': 4, 'need': 4}], 'player': {'circle': 1, 'hp': 10}}
    after = {'requirements': [{'have': 2, 'need': 2}, {'have': 4, 'need': 4}], 'player': {'circle': 1, 'hp': 10}}
    assert requirement_gap(before) == 1
    assert score_transition(before, after).reward > 0.05
    done = score_transition(after, {'requirements': [], 'player': {'circle': 2, 'hp': 10}})
    assert done.reached_target and done.reward == 100

def test_trainable_readout_updates_and_persists_without_changing_connectome():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'policy.json'
        learner = LinearActionLearner(path, epsilon=0, seed=3)
        before = learner.score('look', [1, 0])
        update = learner.update('look', [1, 0], 4)
        assert update['updates'] == 1
        assert learner.score('look', [1, 0]) != before
        restored = LinearActionLearner(path, epsilon=0, seed=3)
        assert restored.updates == 1
        assert restored.score('look', [1, 0]) == learner.score('look', [1, 0])

def test_balanced_rewards_favor_lagging_requirements_and_cap_overtraining():
    def state(a, b):
        return {'player': {'circle': 1, 'hp': 10}, 'requirements': {'circle': 2, 'rows': [
            {'label': 'weapon', 'have': a, 'need': 10},
            {'label': 'survival', 'have': b, 'need': 10}]}}
    before = state(1, 8)
    assert score_transition(before, state(2, 8)).reward > score_transition(before, state(1, 9)).reward
    assert score_transition(state(10, 1), state(11, 1), kill=True).reward < 0
    useful = score_transition(state(10, 1), state(11, 2))
    assert useful.reward > 0 and useful.components['overtraining'] == 0
    assert score_transition(before, before, kill=True).reward == 0
    assert score_transition({}, {'player': {'circle': 1}}, kill=True).reward == 0
    assert requirement_gap({'requirements': [{'have': 'unknown', 'need': 10}]}) is None
    done = {'player': {'circle': 2, 'hp': 10}}
    assert score_transition(done, done).reward == 0
    assert score_transition(before, state(2, 8)).reward == sum(score_transition(before, state(2, 8)).components.values())

def test_pilot_terminal_evidence_and_timeout(tmp_path):
    from dr_adapter.pilot import FlyPilot
    class Transport(MockTransport):
        def read_state(self): return {'player': {'circle': 2, 'hp': 10}}
    transport = Transport()
    pilot = FlyPilot(transport, None, tmp_path / 'target' / 'run.jsonl')
    pilot.run()
    assert pilot.ledger.data['last']['status'] == 'target'
    assert transport.sent == [] and transport.stopped
    expired = FlyPilot(Transport(), None, tmp_path / 'expired' / 'run.jsonl', minutes=0)
    expired.step = 5
    expired.run()
    assert expired.ledger.data['last']['status'] == 'timeout'
    assert expired.t.stopped

def test_pilot_recovers_before_resuming_hunt(tmp_path):
    from dr_adapter.pilot import FlyPilot
    pilot = FlyPilot(MockTransport(), None, tmp_path / 'recovery.jsonl')
    state = {'player': {'circle': 1, 'hp': 30, 'maxHp': 100},
             'room': {'id': 'sewers_1'}, 'combat': {'enemies': [{'name': 'rat'}]}}
    command, reason = pilot.choose(state, None)
    assert command == 'retreat' and allowed_action(command) is not None
    state['combat'] = None
    state['player']['hp'] = 60
    assert pilot.choose(state, None) == ('rest', 'low-health')
    state['player']['hp'] = 85
    assert pilot.choose(state, None)[1] != 'low-health'

def test_curriculum_only_offers_observed_unmet_field_work():
    from dr_adapter.curriculum import field_candidates
    state = {'room': {'id': 'sewers_1'}, 'inventory': [{'id': 'dagger'}],
             'requirements': {'circle': 2, 'rows': [
                 {'label': '1st survival', 'have': 0, 'need': 4},
                 {'label': '1st lore', 'have': 0, 'need': 2}]}}
    assert field_candidates(state) == ['hunt', 'forage', 'perform', 'appraise dagger']
    for command in field_candidates(state):
        action = allowed_action(command.split()[0], configured_command=command)
        assert action is not None and action.command == command
    assert allowed_action('appraise', configured_command='appraise dagger;drop all') is None
    assert allowed_action('appraise', configured_command='buy dagger') is None
    state['combat'] = {'enemies': ['rat']}
    assert field_candidates(state) == []
    state['combat'] = None
    state['requirements']['rows'][0]['have'] = 4
    state['requirements']['rows'][1]['have'] = 2
    assert field_candidates(state) == []
    assert field_candidates({'room': {'id': 'sewers_1'}}) == []

def test_navigation_uses_directed_observed_exits_and_reroutes():
    from dr_adapter.navigation import ObservedMap
    nav = ObservedMap()
    nav.observe({'id': 'square', 'exits': {'n': 'lane', 'e': 'market'}})
    nav.observe({'id': 'lane', 'exits': {'s': 'square', 'w': 'hall_barbarian'}})
    assert nav.next_step('square', 'hall_barbarian') == 'n'
    assert nav.next_step('market', 'square') is None  # no invented reverse edge
    nav.observe({'id': 'lane', 'exits': {'s': 'square'}})
    assert nav.next_step('square', 'hall_barbarian') is None
    assert nav.next_step('square', 'hall_barbarian', explore=True) == 'e'
    assert nav.next_step('square', 'square') is None

def test_equipment_rotation_uses_distinct_owned_lanes_and_live_targets():
    from dr_adapter.curriculum import WEAPONS, equipment_proposal
    state = {'requirements': {'circle': 2, 'rows': [
        {'label': label, 'have': 0, 'need': need} for label, need in
        zip(('1st weapon', '2nd weapon', '3rd weapon', '4th weapon'), (8, 8, 4, 2))]},
        'inventory': [{'id': item} for item in WEAPONS], 'equipment': {'hand': 'dagger'},
        'skills': {skill: {'rank': 0} for skill in WEAPONS.values()}}
    assert equipment_proposal(state) is None
    state['skills']['small_edged']['rank'] = 8
    assert equipment_proposal(state) == 'wield club'
    state['skills']['blunt']['rank'] = 8
    assert equipment_proposal(state) == 'wield staff'
    state['skills']['staff']['rank'] = 4
    assert equipment_proposal(state) == 'wield long_sword'
    state['skills']['medium_edged']['rank'] = 2
    assert equipment_proposal(state) is None
    state['inventory'].append({'id': 'shield_wood'})
    assert equipment_proposal(state) == 'wear shield_wood'
    state['combat'] = {'enemies': ['rat']}
    assert equipment_proposal(state) is None

def test_delayed_outcome_includes_progress_without_new_commands():
    from dr_adapter.outcome import settle_outcome
    now = [0.0]
    initial = {'player': {'hp': 10, 'circle': 1}, 'requirements': [{'have': 0, 'need': 2}]}
    final = {'player': {'hp': 10, 'circle': 1}, 'requirements': [{'have': 1, 'need': 2}]}
    class Transport(MockTransport):
        def read_state(self): return final
    transport = Transport()
    result = settle_outcome(transport, initial, until=2, deadline=5,
                            clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert result == final and now[0] == 2
    assert not transport.sent
    assert score_transition(initial, result).reward > 0
    dead = {'player': {'hp': 0, 'circle': 1}}
    assert settle_outcome(transport, dead, until=9, deadline=10, clock=lambda: now[0]) == dead
    assert now[0] == 2
    settle_outcome(transport, {}, until=9, deadline=10, clock=lambda: now[0])
    assert transport.stopped

def test_purchases_respect_live_stock_budget_and_owned_items():
    from dr_adapter.curriculum import purchase_proposal
    state = {'player': {'silver': 100}, 'inventory': [{'id': 'dagger'}],
             'equipment': {'torso': 'leather'}, 'shopStock': [
                 {'id': 'shield_wood', 'quantity': 1, 'price': 70},
                 {'id': 'club', 'quantity': 1, 'price': 112}]}
    assert purchase_proposal(state) == 'buy shield_wood'
    state['shopStock'][0]['quantity'] = 0
    assert purchase_proposal(state) is None
    state['player']['silver'] = 112
    assert purchase_proposal(state) == 'buy club'
    assert allowed_action('skin', configured_command='skin sewer_rat').command == 'skin sewer_rat'
    assert allowed_action('skin', configured_command='skin rat;drop all') is None
    assert allowed_action('skin', configured_command='skin') is None

def test_bazaar_departure_does_not_reset_outgoing_route(tmp_path):
    from dr_adapter.pilot import FlyPilot
    pilot = FlyPilot(MockTransport(), None, tmp_path / 'market.jsonl')
    state = {'player': {'hp': 10, 'maxHp': 10, 'circle': 1, 'silver': 0},
             'room': {'id': 'bazaar', 'exits': {'w': 'lane'}}, 'shopStock': []}
    assert pilot.choose(state, None) == ('look', 'return-to-hunt')
    assert pilot.choose(state, None) == ('w', 'town-route')

def test_exp_reward_tracks_absorption_once_and_excludes_overleveled_skills():
    from dr_adapter.reward import useful_exp_reward
    before = {'requirements': {'circle': 2, 'rows': [{'label': '1st weapon', 'have': 1, 'need': 8, 'eligible': ['blunt']}]},
              'skills': {'blunt': {'rank': 1, 'exp': 0, 'nextRankExp': 100, 'absorbedExp': 100, 'pooledExp': 50}}}
    after = {'skills': {'blunt': {'absorbedExp': 125, 'pooledExp': 25}}}
    reward = useful_exp_reward(before, after)
    assert 0 < reward <= .1
    assert useful_exp_reward(before, before) == 0
    before['skills']['blunt']['rank'] = 8
    assert useful_exp_reward(before, after) == 0
    assert useful_exp_reward({}, after) == 0

def test_dr_connectome_projection_is_reward_plastic_and_checkpointed():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / 'dr-connectome.npz'
        bridge = FlyBridge(checkpoint=checkpoint)
        bridge.observe(parse_transcript('Location: Arena\nHP: 10/10\nattacking orc'))
        before = bridge.brain.input_projection.copy()
        update = bridge.learn(1.0)
        assert update['updates'] == 1
        assert not np.array_equal(before, bridge.brain.input_projection)
        assert checkpoint.exists()

def test_ready_field_observation_has_activity_and_measured_weight_change():
    bridge = FlyBridge()
    output = bridge.observe(parse_transcript('Location: Field\nHP: 10/10'))
    assert output['drive_mapping'] == 'dr-sensory/2'
    assert output['activity_rms'] > 0
    before = bridge.brain.input_projection.copy()
    update = bridge.learn(.05)
    measured = float(np.sqrt(np.mean((bridge.brain.input_projection - before) ** 2)))
    assert measured > 0 and update['weight_delta_rms'] == measured

def test_analysis_commands_are_explicitly_bounded():
    assert allowed_action('analyze', configured_command='analyze flame').command == 'analyze flame'
    assert allowed_action('analyze', configured_command='analyze arbitrary') is None
    assert allowed_action('analyze', configured_command='analyze flame;drop all') is None

def test_initial_ability_training_is_at_hall_and_narrowly_allowlisted(tmp_path):
    from dr_adapter.pilot import FlyPilot
    pilot = FlyPilot(MockTransport(), None, tmp_path / 'ability.jsonl')
    state = {'player': {'circle': 1, 'hp': 10, 'maxHp': 10, 'guild': 'barbarian', 'abilities': []},
             'room': {'id': 'hall_barbarian', 'exits': {'n': 'lane'}}}
    assert pilot.choose(state, None) == ('learn dragon', 'requirement-ability')
    for verb in ('learn', 'form'):
        assert allowed_action(verb, configured_command=verb + ' dragon').command == verb + ' dragon'
        assert allowed_action(verb, configured_command=verb + ' arbitrary') is None

def test_frozen_mode_never_updates_or_writes_learning_checkpoints(tmp_path):
    from dr_adapter.pilot import FlyPilot
    bridge = FlyBridge(checkpoint=tmp_path / 'projection.npz')
    pilot = FlyPilot(MockTransport(), bridge, tmp_path / 'run.jsonl', mode='frozen')
    original = bridge.brain.input_projection.copy()
    features = bridge.observe(parse_transcript('Location: Field\nHP: 10/10'))['features']
    learning, circuit = pilot.learn_transition('look', features, 100)
    assert learning['enabled'] is False and circuit['weight_delta_rms'] == 0
    assert np.array_equal(original, bridge.brain.input_projection)
    assert not (tmp_path / 'projection.npz').exists()
    assert not (tmp_path / 'fly-policy.json').exists()
    assert pilot.ledger.data['last']['mode'] == 'frozen'

def test_checkpoint_copy_preserves_source_and_records_exact_hashes(tmp_path):
    import hashlib

    from dr_adapter.experiment import copy_checkpoint
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    destination.mkdir()
    bridge = FlyBridge(checkpoint=source / 'connectome.npz')
    bridge.observe(parse_transcript('Location: Field\nHP: 10/10'))
    bridge.learn(.1)
    learner = LinearActionLearner(source / 'policy.json')
    learner.update('look', [1], 1)
    hashes = copy_checkpoint(source, destination)
    for name, digest in hashes.items():
        assert (source / name).read_bytes() == (destination / name).read_bytes()
        assert digest == hashlib.sha256((source / name).read_bytes()).hexdigest()
