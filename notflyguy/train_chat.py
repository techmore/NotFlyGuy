"""Train and honestly evaluate the group-chat behaviour policy.

Evaluation uses leave-sessions-out folds: no message from a held-out session is
ever seen during training, so the reported numbers measure transfer to a new
conversation rather than recall of the training transcripts.

Two models are trained per fold -- context features alone, and context plus
connectome state -- because the interesting question is not "does a linear
model beat guessing" but "does the measured circuit add anything beyond the
obvious surface features". Both answers are reported as measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .chat_policy import ChatPolicy, accuracy, brier_score, majority_baseline, mean_absolute_error, roc_auc
from .chatlog import corpus_stats, default_db_paths, load_all
from .groupchat import build_decision_points, dataset_stats, feature_matrix, feature_names

DEFAULT_OUTPUT = 'data/training/chat-policy-v1'
SILENT = 0.0


def make_folds(points, folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Assign whole sessions to folds so no session spans a split."""
    sessions = sorted({point.session_id for point in points})
    if folds < 2 or len(sessions) < 2:
        return [np.arange(len(points))]
    folds = min(folds, len(sessions))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sessions))
    assignment: dict[str, int] = {session: i % folds for i, session in enumerate(shuffled)}
    buckets = [[] for _ in range(folds)]
    for index, point in enumerate(points):
        buckets[assignment[point.session_id]].append(index)
    return [np.array(bucket, dtype=int) for bucket in buckets if bucket]


def _fold_metrics(y_true, probability, y_length_true, length_pred, train_length_mean) -> dict:
    length_mask = np.isfinite(y_length_true)
    if length_mask.any():
        baseline_pred = np.full(int(length_mask.sum()), train_length_mean)
        length_mae_baseline = mean_absolute_error(y_length_true[length_mask], baseline_pred)
    else:
        length_mae_baseline = float('nan')
    return {
        'n': int(len(y_true)),
        'positives': int(np.sum(y_true > 0.5)),
        'accuracy': accuracy(y_true, probability),
        'majority_baseline': majority_baseline(y_true),
        'roc_auc': roc_auc(y_true, probability),
        'brier': brier_score(y_true, probability),
        'length_mae': mean_absolute_error(y_length_true, length_pred),
        'length_mae_baseline': length_mae_baseline,
    }


def evaluate_cross_validation(points, folds: int = 5, seed: int = 0, l2: float = 1.0,
                              length_l2: float = 10.0, epochs: int = 3000,
                              lr: float = 0.5) -> dict:
    """Return per-fold and aggregate metrics for both feature sets."""
    splits = make_folds(points, folds=folds, seed=seed)
    results = {}
    for label, use_connectome in (('context_only', False), ('context_plus_connectome', True)):
        X, y_respond, y_length = feature_matrix(points, include_connectome=use_connectome)
        names = feature_names(include_connectome=use_connectome)
        fold_rows = []
        oof_probability = np.full(len(points), np.nan)
        oof_length = np.full(len(points), np.nan)
        for held_out in splits:
            train_mask = np.ones(len(points), dtype=bool)
            train_mask[held_out] = False
            if train_mask.sum() == 0 or held_out.size == 0:
                continue
            policy = ChatPolicy(X.shape[1], l2=l2, length_l2=length_l2, epochs=epochs,
                                lr=lr, names=names)
            policy.fit(X[train_mask], y_respond[train_mask], y_length[train_mask])
            probability = policy.predict_proba(X[held_out])
            length_pred = policy.predict_length(X[held_out])
            train_lengths = y_length[train_mask]
            train_length_mean = (float(np.nanmean(train_lengths))
                                 if np.isfinite(train_lengths).any() else 0.0)
            oof_probability[held_out] = probability
            oof_length[held_out] = np.where(np.isfinite(y_length[held_out]),
                                            length_pred, np.nan)
            fold_rows.append(_fold_metrics(y_respond[held_out], probability,
                                           y_length[held_out], length_pred, train_length_mean))
        finite = np.isfinite(oof_probability)
        results[label] = {
            'include_connectome': use_connectome,
            'folds': fold_rows,
            'mean_accuracy': float(np.mean([f['accuracy'] for f in fold_rows])) if fold_rows else float('nan'),
            'mean_majority_baseline': float(np.mean([f['majority_baseline'] for f in fold_rows])) if fold_rows else float('nan'),
            'mean_roc_auc': float(np.nanmean([f['roc_auc'] for f in fold_rows])) if fold_rows else float('nan'),
            'pooled_accuracy': accuracy(y_respond[finite], oof_probability[finite]) if finite.any() else float('nan'),
            'pooled_roc_auc': roc_auc(y_respond[finite], oof_probability[finite]) if finite.any() else float('nan'),
            'pooled_brier': brier_score(y_respond[finite], oof_probability[finite]) if finite.any() else float('nan'),
        }
    return {'folds': len(splits), 'results': results}


def verdict(cv: dict, stats: dict, min_points: int, min_group_points: int,
            min_group_sessions: int = 5, min_connectome_delta: float = 0.005) -> dict:
    """Say plainly what the numbers do and do not establish.

    A feature set only counts as helping when it gains a real margin on every
    fold. Folds that move by exactly zero are not evidence, and a gain that
    coincides with a drop in ranking quality is not evidence either.

    Sufficiency requires distinct group *sessions*, not just messages: the
    evaluation is leave-sessions-out, so many messages from one conversation
    still yield one fold and cannot show transfer to a new group.
    """
    context = cv['results']['context_only']
    full = cv['results']['context_plus_connectome']
    beats_majority = bool(full['mean_accuracy'] > full['mean_majority_baseline'])
    connectome_delta = float(full['mean_accuracy'] - context['mean_accuracy'])
    auc_delta = float(full['mean_roc_auc'] - context['mean_roc_auc'])
    fold_deltas = [
        f_full['accuracy'] - f_ctx['accuracy']
        for f_full, f_ctx in zip(full['folds'], context['folds'])
    ]
    connectome_helps = bool(
        fold_deltas
        and connectome_delta >= min_connectome_delta
        and all(d >= 0 for d in fold_deltas)
        and auc_delta >= 0
    )
    def _mean_of(folds, key):
        values = [f[key] for f in folds if key in f and np.isfinite(f[key])]
        return float(np.mean(values)) if values else float('nan')

    length_mae = _mean_of(full['folds'], 'length_mae')
    length_baseline = _mean_of(full['folds'], 'length_mae_baseline')
    length_measurable = bool(np.isfinite(length_mae) and np.isfinite(length_baseline))
    length_helps = bool(length_measurable and length_mae < length_baseline)
    sufficient = bool(stats['decision_points'] >= min_points
                      and stats['group_decision_points'] >= min_group_points
                      and stats.get('sessions', 0) >= min_group_sessions)
    if not sufficient:
        claim = (
            'NOT ESTABLISHED: the corpus is below the configured sufficiency '
            f"thresholds ({stats['decision_points']} decision points, "
            f"{stats['group_decision_points']} from actual group chats across "
            f"{stats.get('sessions', 0)} session(s); thresholds are {min_points}, "
            f"{min_group_points} and {min_group_sessions}). The trained artifact is "
            'a plumbing check that the pipeline runs end to end, not evidence of '
            'learned group-chat behaviour.'
        )
    elif not beats_majority:
        claim = (
            'NOT ESTABLISHED: held-out accuracy did not exceed the majority-class '
            'baseline on unseen sessions, so the policy is not shown to predict '
            'when to speak.'
        )
    elif connectome_helps:
        claim = (
            'LIMITED: the policy beats the majority baseline on held-out sessions '
            'and connectome features improved every fold over context alone. This '
            'is a behavioural predictor, not a language model, and the circuit is '
            'a truncated motor-centred subgraph.'
        )
    else:
        claim = (
            'PARTIAL: the policy beats the majority baseline on held-out sessions, '
            'but connectome features did not add a real margin over context alone '
            f'({connectome_delta:+.4f} accuracy, {auc_delta:+.4f} AUC), so the '
            'measured circuit is not yet shown to add predictive value for '
            'group-chat behaviour.'
        )
    if length_measurable and not length_helps:
        claim += (
            f' The reply-length head does not beat a constant predictor on held-out '
            f'sessions ({length_mae:.4f} vs {length_baseline:.4f} MAE), so its length '
            'estimate should not be relied on.'
        )
    return {
        'sufficient_data': sufficient,
        'beats_majority_baseline': beats_majority,
        'connectome_adds_predictive_value': connectome_helps,
        'connectome_accuracy_delta': connectome_delta,
        'connectome_auc_delta': auc_delta,
        'min_connectome_delta': min_connectome_delta,
        'per_fold_connectome_delta': [float(d) for d in fold_deltas],
        'accuracy_vs_majority': float(full['mean_accuracy'] - full['mean_majority_baseline']),
        'length_head_beats_constant': length_helps,
        'length_mae': length_mae,
        'length_mae_baseline': length_baseline,
        'claim': claim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the group-chat behaviour policy.')
    parser.add_argument('--db', action='append', help='Hermes database (repeatable)')
    parser.add_argument('--chat-types', default='all',
                        help="comma-separated sessions.chat_type filter, or 'all'")
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--min-points', type=int, default=200)
    parser.add_argument('--min-group-points', type=int, default=50)
    parser.add_argument('--min-group-sessions', type=int, default=5,
                        help='distinct group conversations required; leave-sessions-out '
                             'cannot generalize from one session however many messages it has')
    parser.add_argument('--force', action='store_true', help='overwrite an existing output directory')
    args = parser.parse_args()
    if args.folds < 2:
        parser.error('need at least 2 folds')

    paths = [Path(p) for p in args.db] if args.db else None
    chat_types = None if args.chat_types == 'all' else tuple(t for t in args.chat_types.split(',') if t)
    output = Path(args.output)
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; pass --force to overwrite')

    events = load_all(paths, chat_types=chat_types)
    if not events:
        parser.error('no events loaded; check --db and --chat-types')
    points = build_decision_points(events, seed=0)
    corpus = corpus_stats(events)
    stats = dataset_stats(points)
    cv = evaluate_cross_validation(points, folds=args.folds, seed=0, l2=args.l2)
    outcome = verdict(cv, stats, args.min_points, args.min_group_points,
                      args.min_group_sessions)

    X, y_respond, y_length = feature_matrix(points, include_connectome=True)
    final = ChatPolicy(X.shape[1], l2=args.l2, names=feature_names(include_connectome=True))
    final.fit(X, y_respond, y_length)
    output.mkdir(parents=True, exist_ok=True)
    final.save(output / 'policy.npz')

    report = {
        'policy': 'linear respond + ridge length head over context and connectome features',
        'databases': [str(p) for p in (paths or default_db_paths())],
        'gates': {
            'min_points': args.min_points,
            'min_group_points': args.min_group_points,
            'min_group_sessions': args.min_group_sessions,
        },
        'corpus': corpus,
        'dataset': stats,
        'cross_validation': cv,
        'verdict': outcome,
        'model': final.describe(),
        'limitations': [
            'The connectome is a truncated motor-centred subgraph, not a whole brain.',
            'Drive channels and decision labels are modeling choices, not measured biology.',
            'The policy predicts when and how long to speak; it does not generate language.',
            'Group-chat decision points are far fewer than direct-chat ones; per-group skill is not established.',
            'No message text is stored in this report.',
        ],
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'dataset': stats,
                      'verdict': {k: v for k, v in outcome.items() if k != 'claim'}}, indent=2))
    print(outcome['claim'])


if __name__ == '__main__':
    main()
