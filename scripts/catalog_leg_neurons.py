"""Catalog released leg-motor annotations; no assumed actuator correspondence."""
import json
from collections import Counter
from pathlib import Path
import pyarrow.feather as feather

root = Path(__file__).resolve().parents[1] / 'data' / 'malecns-v1.0'
annotations = feather.read_table(root / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
transmitters = feather.read_table(root / 'body-neurotransmitters-male-cns-v1.0.feather')
nt_by_id = {row['body']: row for row in transmitters.to_pylist()}
selected = []
for row in annotations.to_pylist():
    if row['superclass'] != 'vnc_motor' or row['subclass'] not in ('fl', 'ml', 'hl'):
        continue
    nt = nt_by_id.get(row['bodyId'], {})
    selected.append({key: row[key] for key in ('bodyId', 'type', 'subclass', 'somaSide', 'somaNeuromere')} |
                    {'consensus_nt': nt.get('consensus_nt')})
selected.sort(key=lambda row: row['bodyId'])
result = {'dataset': 'MaleCNS v1.0', 'selection': "superclass=vnc_motor and subclass in fl,ml,hl",
          'count': len(selected), 'counts_by_subclass': dict(Counter(row['subclass'] for row in selected)),
          'limitations': ['Motor annotations are not a verified mapping to FlyBody actuators.',
                          'Circuit connectivity is not yet extracted.',
                          'Transmitter labels do not specify physiological synaptic strength.'],
          'neurons': selected}
(root / 'leg-motor-catalog.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ('count', 'counts_by_subclass', 'limitations')}, indent=2))
