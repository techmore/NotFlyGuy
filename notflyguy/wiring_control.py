"""Does the measured wiring actually matter?

The whole project claims to be connectome-powered, but a claim like that is
empty without a control. This module builds the missing control: a
degree-preserving rewiring of the same graph -- identical neurons, identical
in-degree and out-degree for every neuron, identical synapse-count values,
different pairing -- and re-runs the same evaluation against it.

This mirrors the control in ngxson's fly-llm model card, where a randomly
permuted wiring of the same connectome reached train loss 0.77 / validation
4.27 against 1.02 / 3.84 for the real wiring.

Honest scope: the shuffle preserves the directed degree sequence and the
per-source weight multiset. It does *not* preserve any higher-order structure
(motifs, clustering, spatial arrangement), which is exactly the point -- those
are the candidate things the real wiring might contribute.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .chatlog import default_db_paths, load_all
from .connectome import DEFAULT_ROOT, ConnectomeState
from .groupchat import build_decision_points
from .train_chat import evaluate_cross_validation

DEFAULT_OUTPUT = 'data/training/wiring-control'
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def shuffle_edges(pre, post, seed=0, swaps_per_edge=10) -> dict:
    """Randomly rewire targets while preserving the directed degree sequence.

    Sources never move: only the *targets* of existing edges are permuted, so
    every neuron keeps its exact in-degree and out-degree, and each source
    keeps its exact multiset of outgoing weights. Swaps that would create a
    self-loop or a duplicate edge are rejected.
    """
    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64).copy()
    edges = len(pre)
    rng = np.random.default_rng(seed)
    existing = set(zip(pre.tolist(), post.tolist()))
    attempts = int(swaps_per_edge * edges)
    accepted = 0
    for _ in range(attempts):
        i = int(rng.integers(0, edges))
        j = int(rng.integers(0, edges))
        if i == j:
            continue
        a, b = int(pre[i]), int(post[i])
        c, d = int(pre[j]), int(post[j])
        if a == c or b == d:
            continue
        if a == d or c == b:
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        existing.add((a, d))
        existing.add((c, b))
        post[i] = d
        post[j] = b
        accepted += 1
    return {'post': post, 'attempts': attempts, 'accepted': accepted,
            'acceptance_rate': accepted / attempts if attempts else 0.0}


def _degrees(pre, post, n_nodes):
    out_degree = np.bincount(pre, minlength=n_nodes)
    in_degree = np.bincount(post, minlength=n_nodes)
    return out_degree, in_degree


def build_shuffled_circuit(source_npz, dest_npz, seed=0, swaps_per_edge=10) -> dict:
    """Write a rewired copy of a circuit, preserving node order and node roles."""
    source_npz, dest_npz = Path(source_npz), Path(dest_npz)
    with np.load(source_npz, allow_pickle=False) as graph:
        data = {key: graph[key] for key in graph.files}
    n_nodes = len(data['body_ids'])
    pre, post = data['pre'], data['post']
    out_before, in_before = _degrees(pre, post, n_nodes)

    shuffled = shuffle_edges(pre, post, seed=seed, swaps_per_edge=swaps_per_edge)
    new_post = shuffled['post']
    out_after, in_after = _degrees(pre, new_post, n_nodes)

    if not np.array_equal(out_before, out_after) or not np.array_equal(in_before, in_after):
        raise AssertionError('rewiring changed the degree sequence')
    changed = int(np.sum(new_post != post))

    dest_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload['post'] = new_post
    np.savez_compressed(dest_npz, **payload)

    return {
        'seed': int(seed),
        'dest': str(dest_npz),
        'edges': int(len(pre)),
        'edges_retargeted': changed,
        'edges_retargeted_fraction': changed / len(pre) if len(pre) else 0.0,
        'swaps_accepted': shuffled['accepted'],
        'swap_acceptance_rate': shuffled['acceptance_rate'],
        'degree_sequence_preserved': True,
        'weights_preserved_by_source': True,
    }


def _cv_summary(cv: dict) -> dict:
    full = cv['results']['context_plus_connectome']
    context = cv['results']['context_only']
    return {
        'mean_accuracy': full['mean_accuracy'],
        'mean_majority_baseline': full['mean_majority_baseline'],
        'mean_roc_auc': full['mean_roc_auc'],
        'pooled_accuracy': full['pooled_accuracy'],
        'pooled_roc_auc': full['pooled_roc_auc'],
        'context_only_roc_auc': context['mean_roc_auc'],
    }


def evaluate(source_npz, events, shuffle_seeds=SHUFFLE_SEEDS, folds=5, l2=1.0,
             swaps_per_edge=10, workdir=None, points=None,
             min_points=200, min_sessions=5) -> dict:
    """Compare the real circuit against degree-preserving shuffles.

    The comparison refuses to make a claim on a corpus too small to support one.
    Leave-sessions-out AUC on a handful of decision points is quantised and can
    order real above shuffled by chance alone.
    """
    source_npz = Path(source_npz)
    workdir = Path(workdir or source_npz.parent / 'wiring-control')
    root = source_npz.parent

    real_brain = ConnectomeState(root, seed=0, circuit_file=source_npz.name)
    real_points = build_decision_points(events, brain=real_brain) if points is None else points
    real_cv = evaluate_cross_validation(real_points, folds=folds, seed=0, l2=l2)
    real = _cv_summary(real_cv)

    from .groupchat import dataset_stats
    stats = dataset_stats(real_points)
    sufficient = bool(stats['decision_points'] >= min_points
                      and stats['sessions'] >= min_sessions)

    shuffles = []
    for seed in shuffle_seeds:
        dest = workdir / f'shuffled-{seed}.npz'
        meta = build_shuffled_circuit(source_npz, dest, seed=seed,
                                      swaps_per_edge=swaps_per_edge)
        brain = ConnectomeState(root, seed=0, circuit_file=dest.resolve())
        shuffled_points = build_decision_points(events, brain=brain)
        summary = _cv_summary(evaluate_cross_validation(shuffled_points, folds=folds,
                                                        seed=0, l2=l2))
        summary.update({'seed': seed, 'edges_retargeted': meta['edges_retargeted'],
                        'edges_retargeted_fraction': meta['edges_retargeted_fraction']})
        shuffles.append(summary)

    aucs = np.array([s['mean_roc_auc'] for s in shuffles], dtype=float)
    accs = np.array([s['mean_accuracy'] for s in shuffles], dtype=float)
    beats_all = bool(real['mean_roc_auc'] > aucs.max()) if len(aucs) else False
    within = bool(len(aucs) and aucs.min() <= real['mean_roc_auc'] <= aucs.max())

    if not sufficient:
        claim = (
            'NOT ESTABLISHED: the corpus is too small to support a wiring claim '
            f"({stats['decision_points']} decision points across {stats['sessions']} "
            f"session(s); thresholds are {min_points} and {min_sessions}). The AUC "
            'ordering below is reported for the record and must not be read as '
            'evidence either way.'
        )
    elif not len(aucs):
        claim = 'NOT ESTABLISHED: no shuffle controls were evaluated.'
    elif beats_all:
        claim = (
            'SUPPORTED: the real wiring scored a higher held-out ROC AUC than every '
            f'degree-preserving shuffle ({real["mean_roc_auc"]:.4f} vs best shuffled '
            f'{aucs.max():.4f}). On this task the measured wiring carries information '
            'that the degree sequence alone does not.'
        )
    else:
        claim = (
            'NOT DEMONSTRATED: the real wiring did not beat every degree-preserving '
            f'shuffle (real AUC {real["mean_roc_auc"]:.4f} vs shuffled '
            f'{aucs.min():.4f}-{aucs.max():.4f}, mean {aucs.mean():.4f}). The measured '
            'wiring is therefore not shown to contribute anything beyond its degree '
            'sequence on this task.'
        )

    verdict = {
        'sufficient_data': sufficient,
        'decision_points': int(stats['decision_points']),
        'sessions': int(stats['sessions']),
        'min_points': min_points,
        'min_sessions': min_sessions,
        'fold_vs_pooled_auc_gap': float(real['mean_roc_auc'] - real['pooled_roc_auc']),
        'real_beats_every_shuffle': beats_all,
        'real_within_shuffled_range': within,
        'auc_margin_over_best_shuffle': float(real['mean_roc_auc'] - aucs.max()) if len(aucs) else None,
        'shuffled_auc_mean': float(aucs.mean()) if len(aucs) else None,
        'shuffled_auc_std': float(aucs.std()) if len(aucs) else None,
        'shuffled_auc_min': float(aucs.min()) if len(aucs) else None,
        'shuffled_auc_max': float(aucs.max()) if len(aucs) else None,
        'shuffled_accuracy_mean': float(accs.mean()) if len(accs) else None,
        'claim': claim,
    }
    return {
        'question': 'Does the measured wiring beat a degree-preserving shuffle?',
        'circuit': str(source_npz),
        'folds': folds,
        'l2': l2,
        'shuffle_seeds': list(shuffle_seeds),
        'real': real,
        'shuffles': shuffles,
        'verdict': verdict,
        'limitations': [
            'Degree sequence and per-source weight multiset are preserved; motifs and '
            'spatial structure are not, which is the point of the control.',
            'The readout is a small linear model over 8 aggregate connectome features; '
            'a richer readout could extract more from the real wiring.',
            'Decision points come mostly from direct chats, not group chats.',
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit', default=str(DEFAULT_ROOT / 'motor-circuit.npz'))
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--chat-types', default='all',
                        help="comma-separated sessions.chat_type filter, or 'all'")
    parser.add_argument('--shuffles', type=int, default=len(SHUFFLE_SEEDS))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--swaps-per-edge', type=int, default=10)
    parser.add_argument('--min-points', type=int, default=200)
    parser.add_argument('--min-sessions', type=int, default=5)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; pass --force to overwrite')

    chat_types = None if args.chat_types == 'all' else tuple(
        t for t in args.chat_types.split(',') if t)
    events = load_all(None, chat_types=chat_types)
    if not events:
        parser.error('no events loaded')

    report = evaluate(Path(args.circuit), events,
                      shuffle_seeds=tuple(range(args.shuffles)),
                      folds=args.folds, l2=args.l2,
                      swaps_per_edge=args.swaps_per_edge,
                      min_points=args.min_points, min_sessions=args.min_sessions)
    report['databases'] = [str(p) for p in default_db_paths()]
    report['events'] = len(events)
    report['chat_types'] = args.chat_types

    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'real': report['real'], 'verdict': report['verdict']}, indent=2))
    print(report['verdict']['claim'])


if __name__ == '__main__':
    main()
