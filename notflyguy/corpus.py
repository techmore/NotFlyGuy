"""Corpus readiness: how much real group-chat data exists and what it takes to train.

The group-chat policy is data-limited, not model-limited. This module answers
the only question that matters for scheduling work: *are we there yet, and if
not, how fast are we getting there?*

The gate deliberately requires distinct group **sessions**, not just messages.
Evaluation is leave-sessions-out, so a thousand messages from one conversation
still yields one fold and cannot show that a policy transfers to a new group.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from .chatlog import default_db_paths, load_all
from .groupchat import build_decision_points, dataset_stats

# Thresholds for making a group-specific claim. These are choices, not laws:
# 5 sessions is roughly the minimum that gives leave-one-session-out any
# stability, and 50 decision points keeps the fold count from being dominated
# by a single message.
DEFAULT_GATES = {
    'group_sessions': 5,
    'group_decision_points': 50,
    'group_participants': 3,
}

RECENT_WEEKS = 3


def _check(have: int, need: int) -> dict:
    return {'have': int(have), 'need': int(need), 'met': bool(have >= need),
            'remaining': max(0, int(need) - int(have))}


def readiness(stats: dict, corpus: dict, gates: dict | None = None) -> dict:
    """Compare measured corpus size against the gates. Pure function."""
    merged = dict(DEFAULT_GATES)
    if gates:
        merged.update(gates)
    checks = {
        'group_sessions': _check(stats.get('sessions', 0), merged['group_sessions']),
        'group_decision_points': _check(stats.get('group_decision_points', 0),
                                        merged['group_decision_points']),
        'group_participants': _check(corpus.get('participants', 0), merged['group_participants']),
    }
    blockers = [f'{name}: {c["have"]}/{c["need"]}' for name, c in checks.items() if not c['met']]
    return {
        'ready': not blockers,
        'gates': merged,
        'checks': checks,
        'blockers': blockers,
    }


def weekly_counts(points) -> dict:
    """Group decision points per ISO week, keyed ``YYYY-Www``."""
    counts: Counter = Counter()
    for point in points:
        stamp = datetime.fromtimestamp(point.timestamp, timezone.utc)
        counts[stamp.strftime('%G-W%V')] += 1
    return dict(sorted(counts.items()))


def project(weekly: dict, need: int, have: int, weeks: int = RECENT_WEEKS) -> dict:
    """Naive linear projection from the most recent weeks. Not a forecast."""
    ordered = list(weekly.values())[-weeks:]
    if not ordered:
        return {'rate_per_week': 0.0, 'weeks_to_gate': None,
                'basis': 'no group decision points recorded'}
    rate = sum(ordered) / len(ordered)
    remaining = max(0, need - have)
    if rate <= 0:
        return {'rate_per_week': 0.0, 'weeks_to_gate': None,
                'basis': f'no new group messages in the last {len(ordered)} active week(s)'}
    return {'rate_per_week': round(rate, 2),
            'weeks_to_gate': round(remaining / rate, 1) if remaining else 0.0,
            'basis': f'mean of the last {len(ordered)} recorded week(s); not a forecast'}


def status(events=None, points=None, all_points=None, gates=None, db_paths=None,
           chat_types=('group',)) -> dict:
    """Full readiness report.

    ``events``/``points``/``all_points`` can be injected so the report is
    testable without touching a real database. When ``events`` is injected and
    ``all_points`` is not, the all-source comparison is reported as unavailable
    rather than quietly reading the developer's own history.
    """
    group_events = load_all(db_paths, chat_types=chat_types) if events is None else events
    group_points = build_decision_points(group_events, seed=0) if points is None else points
    if all_points is None and events is None:
        all_points = build_decision_points(load_all(db_paths, chat_types=None), seed=0)

    from .chatlog import corpus_stats
    corpus = corpus_stats(group_events)
    stats = dataset_stats(group_points)
    weekly = weekly_counts(group_points)
    verdict = readiness(stats, corpus, gates)

    responded = int(stats['responded'])
    total = int(stats['decision_points'])
    if all_points is None:
        all_sources = {'decision_points': None, 'group_share': None,
                       'note': 'not computed for an injected corpus'}
    else:
        all_sources = {
            'decision_points': len(all_points),
            'group_share': (total / len(all_points)) if all_points else 0.0,
            'note': '',
        }
    return {
        'group': {
            'corpus': corpus,
            'dataset': stats,
            'by_source': dict(Counter(e.source or 'unknown' for e in group_events)),
            'weekly_decision_points': weekly,
        },
        'all_sources': all_sources,
        'label_balance': {
            'responded': responded,
            'silent': total - responded,
            'respond_rate': stats['respond_rate'],
            'majority_baseline': stats['majority_class_rate'],
            'note': ('Group-only labels are far less skewed than the all-source mix, '
                     'which is why the group corpus is worth collecting separately.'),
        },
        'projection': project(weekly, verdict['gates']['group_decision_points'], total),
        'readiness': verdict,
    }


def summarise(report: dict) -> str:
    group = report['group']['dataset']
    corpus = report['group']['corpus']
    check = report['readiness']['checks']
    lines = [
        f"Group-chat corpus: {group['decision_points']} decision points "
        f"from {group['sessions']} session(s), {corpus['participants']} participant(s)",
        f"  sessions          {check['group_sessions']['have']}/{check['group_sessions']['need']}",
        f"  decision points   {check['group_decision_points']['have']}/{check['group_decision_points']['need']}",
        f"  participants      {check['group_participants']['have']}/{check['group_participants']['need']}",
        f"  labels            {report['label_balance']['responded']} reply / "
        f"{report['label_balance']['silent']} silent "
        f"({report['label_balance']['respond_rate']:.0%} reply)",
        f"  weekly            {report['group']['weekly_decision_points']}",
        f"  rate              ~{report['projection']['rate_per_week']}/week",
    ]
    share = report['all_sources'].get('group_share')
    if share is not None:
        lines.append(f"  group share of all decision points: {share:.1%}")
    if report['readiness']['ready']:
        lines.append('READY: all gates met; the trainer will now make a group-specific claim.')
    else:
        weeks = report['projection']['weeks_to_gate']
        eta = f'~{weeks} week(s) at the recent rate' if weeks is not None else 'no recent activity'
        lines.append(f'NOT READY: {", ".join(report["readiness"]["blockers"])} ({eta})')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Report group-chat corpus readiness.')
    parser.add_argument('--db', action='append', help='Hermes state.db (repeatable)')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON')
    parser.add_argument('--group-sessions', type=int, default=DEFAULT_GATES['group_sessions'])
    parser.add_argument('--group-points', type=int, default=DEFAULT_GATES['group_decision_points'])
    args = parser.parse_args()
    paths = [__import__('pathlib').Path(p) for p in args.db] if args.db else None
    report = status(db_paths=paths, gates={
        'group_sessions': args.group_sessions,
        'group_decision_points': args.group_points,
    })
    report['databases'] = [str(p) for p in (paths or default_db_paths())]
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(summarise(report))


if __name__ == '__main__':
    main()
