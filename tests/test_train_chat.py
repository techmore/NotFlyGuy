"""Cross-validation splits, trainer plumbing, and the honesty gate."""
import numpy as np

from notflyguy.chat_policy import ChatPolicy
from notflyguy.train_chat import evaluate_cross_validation, make_folds, verdict


def _synthetic_points(count=240, sessions=12, seed=0, signal=3.0):
    """Decision-point lookalikes with a learnable relationship."""
    from notflyguy.connectome import ConnectomeState
    from notflyguy.groupchat import CONTEXT_NAMES, DecisionPoint

    rng = np.random.default_rng(seed)
    context = rng.normal(size=(count, len(CONTEXT_NAMES)))
    connectome = rng.normal(size=(count, len(ConnectomeState.feature_names())))
    logits = signal * context[:, 0] + 2.0 * connectome[:, 3]
    respond = logits > 0
    points = []
    for index in range(count):
        points.append(DecisionPoint(
            session_id=f'session-{index % sessions}',
            group_id=f'group-{index % sessions}',
            sender_id=f'sender-{index % 4}',
            timestamp=float(index),
            index=index,
            context=tuple(context[index]),
            connectome=tuple(connectome[index]),
            respond=bool(respond[index]),
            reply_length=0.5 if respond[index] else -1.0,
            addressed=False,
            is_group=True,
        ))
    return points


def test_folds_never_split_a_session():
    points = _synthetic_points()
    splits = make_folds(points, folds=4, seed=0)
    assert len(splits) == 4
    assigned = np.concatenate(splits)
    assert sorted(assigned.tolist()) == list(range(len(points)))
    for split in splits:
        sessions = {points[i].session_id for i in split}
        for other in splits:
            if other is split:
                continue
            assert not sessions & {points[i].session_id for i in other}


def test_folds_are_deterministic():
    points = _synthetic_points()
    first = [s.tolist() for s in make_folds(points, folds=3, seed=7)]
    second = [s.tolist() for s in make_folds(points, folds=3, seed=7)]
    assert first == second


def test_single_fold_falls_back_gracefully():
    points = _synthetic_points(count=10, sessions=1)
    assert len(make_folds(points, folds=5, seed=0)) == 1


def test_cross_validation_reports_both_feature_sets():
    points = _synthetic_points()
    cv = evaluate_cross_validation(points, folds=4, seed=0)
    assert set(cv['results']) == {'context_only', 'context_plus_connectome'}
    for result in cv['results'].values():
        assert len(result['folds']) == 4
        assert np.isfinite(result['mean_accuracy'])
        assert 0.0 <= result['pooled_accuracy'] <= 1.0
        assert result['pooled_roc_auc'] > 0.5


def test_cross_validation_learns_a_real_signal():
    points = _synthetic_points(signal=4.0)
    cv = evaluate_cross_validation(points, folds=4, seed=0, l2=0.01)
    full = cv['results']['context_plus_connectome']
    assert full['mean_accuracy'] > full['mean_majority_baseline']


def test_verdict_refuses_to_claim_success_on_a_tiny_corpus():
    points = _synthetic_points(count=40, sessions=5, signal=4.0)
    from notflyguy.groupchat import dataset_stats
    cv = evaluate_cross_validation(points, folds=3, seed=0, l2=0.01)
    outcome = verdict(cv, dataset_stats(points), min_points=200, min_group_points=50)
    assert outcome['sufficient_data'] is False
    assert outcome['claim'].startswith('NOT ESTABLISHED')
    assert 'plumbing check' in outcome['claim']


def test_verdict_says_partial_when_connectome_adds_nothing():
    stats = {'decision_points': 5000, 'group_decision_points': 500, 'sessions': 6}
    cv = {'results': {
        'context_only': {'mean_accuracy': 0.80, 'mean_majority_baseline': 0.60,
                         'mean_roc_auc': 0.70,
                         'folds': [{'accuracy': 0.80}, {'accuracy': 0.80}]},
        'context_plus_connectome': {'mean_accuracy': 0.80, 'mean_majority_baseline': 0.60,
                                    'mean_roc_auc': 0.70,
                                    'folds': [{'accuracy': 0.80}, {'accuracy': 0.80}]},
    }}
    outcome = verdict(cv, stats, min_points=200, min_group_points=50)
    assert outcome['sufficient_data'] is True
    assert outcome['beats_majority_baseline'] is True
    assert outcome['connectome_adds_predictive_value'] is False
    assert outcome['claim'].startswith('PARTIAL')


def test_verdict_ignores_a_noise_sized_connectome_gain():
    """Most folds moving by zero, one by ~1%, and AUC down is not evidence."""
    stats = {'decision_points': 5000, 'group_decision_points': 500, 'sessions': 6}
    cv = {'results': {
        'context_only': {'mean_accuracy': 0.8571, 'mean_majority_baseline': 0.8571,
                         'mean_roc_auc': 0.6519,
                         'folds': [{'accuracy': 0.85}] * 4 + [{'accuracy': 0.85}]},
        'context_plus_connectome': {'mean_accuracy': 0.8594, 'mean_majority_baseline': 0.8571,
                                    'mean_roc_auc': 0.6437,
                                    'folds': [{'accuracy': 0.85}] * 4 + [{'accuracy': 0.8616}]},
    }}
    outcome = verdict(cv, stats, min_points=200, min_group_points=50)
    assert outcome['connectome_adds_predictive_value'] is False
    assert outcome['claim'].startswith('PARTIAL')


def test_verdict_accepts_a_real_connectome_gain():
    stats = {'decision_points': 5000, 'group_decision_points': 500, 'sessions': 6}
    cv = {'results': {
        'context_only': {'mean_accuracy': 0.70, 'mean_majority_baseline': 0.60,
                         'mean_roc_auc': 0.70,
                         'folds': [{'accuracy': 0.70}, {'accuracy': 0.70}]},
        'context_plus_connectome': {'mean_accuracy': 0.78, 'mean_majority_baseline': 0.60,
                                    'mean_roc_auc': 0.82,
                                    'folds': [{'accuracy': 0.78}, {'accuracy': 0.78}]},
    }}
    outcome = verdict(cv, stats, min_points=200, min_group_points=50)
    assert outcome['connectome_adds_predictive_value'] is True
    assert outcome['claim'].startswith('LIMITED')


def test_verdict_calls_out_failed_transfer():
    stats = {'decision_points': 5000, 'group_decision_points': 500, 'sessions': 6}
    cv = {'results': {
        'context_only': {'mean_accuracy': 0.55, 'mean_majority_baseline': 0.85,
                         'mean_roc_auc': 0.50, 'folds': [{'accuracy': 0.55}]},
        'context_plus_connectome': {'mean_accuracy': 0.60, 'mean_majority_baseline': 0.85,
                                    'mean_roc_auc': 0.55, 'folds': [{'accuracy': 0.60}]},
    }}
    outcome = verdict(cv, stats, min_points=200, min_group_points=50)
    assert outcome['beats_majority_baseline'] is False
    assert 'did not exceed' in outcome['claim']


def test_verdict_never_hides_a_bad_length_head():
    stats = {'decision_points': 5000, 'group_decision_points': 500, 'sessions': 6}
    fold = {'accuracy': 0.9, 'length_mae': 0.4, 'length_mae_baseline': 0.1}
    cv = {'results': {
        'context_only': {'mean_accuracy': 0.9, 'mean_majority_baseline': 0.6,
                         'mean_roc_auc': 0.8, 'folds': [fold]},
        'context_plus_connectome': {'mean_accuracy': 0.9, 'mean_majority_baseline': 0.6,
                                    'mean_roc_auc': 0.8, 'folds': [fold]},
    }}
    outcome = verdict(cv, stats, min_points=200, min_group_points=50)
    assert outcome['length_head_beats_constant'] is False
    assert 'constant predictor' in outcome['claim']


def test_verdict_requires_distinct_group_sessions():
    """Many messages from one conversation cannot support leave-sessions-out."""
    cv = {'results': {
        'context_only': {'mean_accuracy': 0.9, 'mean_majority_baseline': 0.6,
                         'mean_roc_auc': 0.9, 'folds': [{'accuracy': 0.9}]},
        'context_plus_connectome': {'mean_accuracy': 0.95, 'mean_majority_baseline': 0.6,
                                    'mean_roc_auc': 0.95, 'folds': [{'accuracy': 0.95}]},
    }}
    one_session = {'decision_points': 5000, 'group_decision_points': 5000, 'sessions': 1}
    outcome = verdict(cv, one_session, min_points=200, min_group_points=50)
    assert outcome['sufficient_data'] is False
    assert 'session' in outcome['claim']

    many_sessions = dict(one_session, sessions=8)
    assert verdict(cv, many_sessions, min_points=200,
                   min_group_points=50)['sufficient_data'] is True


def test_trainer_is_importable_without_mujoco():
    import notflyguy.train_chat as module
    assert callable(module.main)


def test_fitted_policy_predicts_length_for_every_row():
    points = _synthetic_points()
    from notflyguy.groupchat import feature_matrix
    X, y_respond, y_length = feature_matrix(points)
    policy = ChatPolicy(X.shape[1], l2=0.1).fit(X, y_respond, y_length)
    assert np.isfinite(policy.predict_proba(X)).all()
    assert np.isfinite(policy.predict_length(X)).all()
