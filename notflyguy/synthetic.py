"""A small random stand-in for the measured circuit, for tests and smoke runs.

The real circuit is derived from a ~1 GB MaleCNS download
(``scripts/bootstrap_data.py``). Nothing produced here is measured biology: the
graph is random, the "leg motor neurons" are labels assigned round-robin, and
the only thing this stand-in shares with the real artifact is its **file
format** -- the same ``motor-circuit.npz`` keys, the same
``leg-motor-catalog.json`` fields, and a ``motor-circuit.json`` summary whose
``synthetic`` flag is ``true`` so nothing downstream can mistake it for data.

Use it to run the test suite on a fresh clone, or to start the dashboard
without the download::

    python -m notflyguy.synthetic --root /tmp/fake-circuit
    NOTFLYGUY_DATA_ROOT=/tmp/fake-circuit python -m notflyguy.server --no-body
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paths import CATALOG_FILE, CIRCUIT_FILE, SUMMARY_FILE

# The real catalog holds 381 annotated leg motor neurons; the readout's frozen
# parameter count (3107) follows from that, so the default matches it.
DEFAULT_MOTOR_NEURONS = 381
DEFAULT_UPSTREAM_NEURONS = 256
DEFAULT_EDGES_PER_NODE = 12

LEGS = (('T1', 'L', 'fl'), ('T1', 'R', 'fl'), ('T2', 'L', 'ml'),
        ('T2', 'R', 'ml'), ('T3', 'L', 'hl'), ('T3', 'R', 'hl'))


def synthetic_catalog(n_motor: int = DEFAULT_MOTOR_NEURONS, first_id: int = 10_000) -> dict:
    """Catalog shaped like ``scripts/catalog_leg_neurons.py`` output."""
    neurons = []
    for index in range(n_motor):
        neuromere, side, subclass = LEGS[index % len(LEGS)]
        neurons.append({'bodyId': first_id + index, 'type': f'synthetic-{subclass}-{index}',
                        'subclass': subclass, 'somaSide': side, 'somaNeuromere': neuromere,
                        'consensus_nt': None})
    counts: dict[str, int] = {}
    for neuron in neurons:
        counts[neuron['subclass']] = counts.get(neuron['subclass'], 0) + 1
    return {
        'dataset': 'synthetic stand-in (not MaleCNS)',
        'synthetic': True,
        'selection': 'labels assigned round-robin over six legs; no biology',
        'count': len(neurons),
        'counts_by_subclass': counts,
        'limitations': ['Synthetic labels. Not a measured catalog.'],
        'neurons': neurons,
    }


def synthetic_graph(n_motor: int = DEFAULT_MOTOR_NEURONS,
                    n_upstream: int = DEFAULT_UPSTREAM_NEURONS,
                    edges_per_node: int = DEFAULT_EDGES_PER_NODE, seed: int = 0) -> dict:
    """Random sparse digraph in the ``motor-circuit.npz`` layout.

    Every motor neuron receives at least one upstream edge, no node has a
    self-loop, and there are no duplicate edges, so the stand-in satisfies the
    same invariants the wiring control relies on.
    """
    if n_motor < 1 or n_upstream < 1:
        raise ValueError('need at least one motor and one upstream neuron')
    rng = np.random.default_rng(seed)
    motors = np.arange(10_000, 10_000 + n_motor, dtype=np.int64)
    upstream = np.arange(20_000, 20_000 + n_upstream, dtype=np.int64)
    nodes = np.unique(np.concatenate([motors, upstream]))
    n_nodes = len(nodes)
    motor_idx = np.searchsorted(nodes, motors)
    upstream_idx = np.searchsorted(nodes, upstream)

    edges: set[tuple[int, int]] = set()
    # Guaranteed upstream -> motor coverage.
    for i, target in enumerate(motor_idx):
        source = upstream_idx[i % n_upstream]
        edges.add((int(source), int(target)))
    # Random fan-out from every node, skipping self-loops.
    for source in range(n_nodes):
        targets = rng.choice(n_nodes, size=min(edges_per_node, n_nodes - 1), replace=False)
        for target in targets:
            if int(target) != source:
                edges.add((source, int(target)))
    ordered = sorted(edges)
    pre = np.array([e[0] for e in ordered], dtype=np.int64)
    post = np.array([e[1] for e in ordered], dtype=np.int64)
    counts = rng.geometric(0.15, size=len(ordered)).astype(np.int64)
    return {'body_ids': nodes, 'pre': pre, 'post': post, 'synapse_counts': counts,
            'motor_indices': motor_idx, 'upstream_indices': upstream_idx}


def write_synthetic_circuit(root, n_motor: int = DEFAULT_MOTOR_NEURONS,
                            n_upstream: int = DEFAULT_UPSTREAM_NEURONS,
                            edges_per_node: int = DEFAULT_EDGES_PER_NODE,
                            seed: int = 0) -> dict:
    """Write the three files the package loads; return the summary."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    catalog = synthetic_catalog(n_motor)
    graph = synthetic_graph(n_motor, n_upstream, edges_per_node, seed)
    np.savez_compressed(root / CIRCUIT_FILE, **graph)
    (root / CATALOG_FILE).write_text(json.dumps(catalog, indent=2) + '\n')
    summary = {
        'dataset': 'synthetic stand-in (not MaleCNS)',
        'synthetic': True,
        'seed': int(seed),
        'nodes': int(len(graph['body_ids'])),
        'edges': int(len(graph['pre'])),
        'motor_neurons': int(n_motor),
        'selection': 'random graph with the motor-circuit.npz layout; no biology',
        'limitations': ['Random wiring. Any result computed on it says nothing about the fly.'],
    }
    (root / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Write a synthetic circuit so the package can run without the '
                    'MaleCNS download. The result is random, not measured.')
    parser.add_argument('--root', required=True, help='directory to write into')
    parser.add_argument('--motor', type=int, default=DEFAULT_MOTOR_NEURONS)
    parser.add_argument('--upstream', type=int, default=DEFAULT_UPSTREAM_NEURONS)
    parser.add_argument('--edges-per-node', type=int, default=DEFAULT_EDGES_PER_NODE)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    summary = write_synthetic_circuit(args.root, args.motor, args.upstream,
                                      args.edges_per_node, args.seed)
    print(json.dumps(summary, indent=2))
    print(f'Point NOTFLYGUY_DATA_ROOT at {Path(args.root).resolve()} to use it.')


if __name__ == '__main__':
    main()
