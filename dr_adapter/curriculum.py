"""Candidate field activities from observed Barbarian circle requirements.

Only offers reversible activities available at the observed location. The
learned readout chooses among these candidates; this module does not send.
"""
from .reward import requirement_rows

# Curated local-game item facts (data/items.js). Distinct weapon skill lanes.
WEAPONS = {'dagger': 'small_edged', 'club': 'blunt', 'staff': 'staff', 'long_sword': 'medium_edged'}
ARMOR = {'leather': 'torso', 'shield_wood': 'shield', 'leather_boots': 'feet',
         'leather_sleeves': 'arms', 'leather_pants': 'legs'}

def purchase_proposal(state):
    owned = {item.get('id') for item in state.get('inventory', [])}
    owned.update((state.get('equipment') or {}).values())
    silver = (state.get('player') or {}).get('silver')
    if not isinstance(silver, (int, float)): return None
    stock = {item.get('id'): item for item in state.get('shopStock', [])}
    for item in ('dagger', 'leather', 'shield_wood', 'club', 'staff', 'long_sword',
                 'leather_boots', 'leather_sleeves', 'leather_pants'):
        offer = stock.get(item, {})
        price = offer.get('price')
        if item not in owned and offer.get('quantity', 0) > 0 and isinstance(price, (int, float)) and 0 <= price <= silver:
            return 'buy ' + item
    return None

def equipment_proposal(state, target=2):
    """Equip owned protection, then favor the least-complete weapon lane.

    Targets come from authoritative Nth-weapon rows, assigned by descending
    rank among the four curated lanes. Purchases are handled separately.
    """
    if state.get('combat'): return None
    rows = requirement_rows(state, target)
    if rows is None: return None
    inventory = {item.get('id') for item in state.get('inventory', [])}
    equipped = state.get('equipment') or {}
    for item, slot in ARMOR.items():
        if item in inventory and not equipped.get(slot): return 'wear ' + item
    needs = [rows.get(label, (0, 0))[1] for label in ('1st weapon', '2nd weapon', '3rd weapon', '4th weapon')]
    if not all(needs): return None
    skills = state.get('skills', {})
    if not all(isinstance(skills.get(skill, {}).get('rank'), (int, float)) for skill in WEAPONS.values()): return None
    ordered = sorted(WEAPONS, key=lambda item: -skills[WEAPONS[item]]['rank'])
    candidates = []
    for item, need in zip(ordered, needs):
        rank = skills[WEAPONS[item]]['rank']
        if rank < need and (item in inventory or equipped.get('hand') == item):
            candidates.append((rank / need, item))
    if not candidates: return None
    item = min(candidates, key=lambda pair: pair[0])[1]
    return None if equipped.get('hand') == item else 'wield ' + item

def field_candidates(state, target=2):
    rows = requirement_rows(state, target)
    if rows is None or state.get('combat'): return []
    room = (state.get('room') or {}).get('id', '')
    missing = {label for label, (have, need) in rows.items() if have < need}
    choices = []
    if any(label.endswith('survival') for label in missing) and room.startswith('sewers_'):
        choices.extend(('hunt', 'forage'))
    if any(label.endswith('lore') for label in missing):
        choices.append('perform')
        # IDs originate in the player's inventory, never free-form input.
        for item in state.get('inventory', []):
            item_id = item.get('id')
            if isinstance(item_id, str) and item_id and all(c.isascii() and (c.isalnum() or c == '_') for c in item_id):
                choices.append('appraise ' + item_id)
                break
        if room in ('temple', 'temple_row', 'academy'): choices.append('study')
    return choices
