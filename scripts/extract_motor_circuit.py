"""Extract a reproducible motor-centered subgraph from released MaleCNS edges."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

root = Path(__file__).resolve().parents[1] / 'data' / 'malecns-v1.0'
catalog = json.loads((root / 'leg-motor-catalog.json').read_text())
motors = np.array([n['bodyId'] for n in catalog['neurons']], dtype=np.int64)
edges = feather.read_table(root / 'connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather',
                          columns=['body_pre', 'body_post', 'weight'])
pre, post, weight = (edges[k].to_numpy() for k in edges.column_names)
incoming = np.isin(post, motors) & ~np.isin(pre, motors)
ids, inverse = np.unique(pre[incoming], return_inverse=True)
strength = np.bincount(inverse, weights=weight[incoming])
# Deterministic tie-break: body ID ascending.
upstream = ids[np.lexsort((ids, -strength))[:1024]]
nodes = np.unique(np.concatenate([motors, upstream]))
retained = np.isin(pre, nodes) & np.isin(post, nodes)
nt = {r['body']: r['consensus_nt'] for r in feather.read_table(
    root / 'body-neurotransmitters-male-cns-v1.0.feather').to_pylist()}
np.savez_compressed(root / 'motor-circuit.npz', body_ids=nodes,
                    pre=np.searchsorted(nodes, pre[retained]),
                    post=np.searchsorted(nodes, post[retained]),
                    synapse_counts=weight[retained], motor_indices=np.searchsorted(nodes, motors),
                    upstream_indices=np.searchsorted(nodes, upstream))
summary = {'dataset': 'MaleCNS v1.0 traced-only', 'nodes': len(nodes),
           'edges': int(retained.sum()), 'motor_neurons': len(motors),
           'selection': 'All 381 cataloged leg motor neurons plus 1024 strongest direct presynaptic neurons by summed motor-input synapse count; all induced edges retained.',
           'body_ids': nodes.tolist(), 'consensus_nt': [nt.get(int(n)) for n in nodes],
           'source_manifest': json.loads((root / 'manifest.json').read_text()),
           'limitations': ['Truncated motor-centered circuit, not a full brain.',
                          'Synapse counts are anatomical counts, not physiological weights.',
                          'Sensory injection and motor projections remain modeling choices.']}
(root / 'motor-circuit.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('dataset','nodes','edges','motor_neurons','selection')}))
