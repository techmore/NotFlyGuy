# Group-chat policy: design and measured results

This document records what was built, what was measured, and what the numbers
do **not** support. Everything below comes from `data/training/chat-policy-v1/report.json`,
regenerated with `python -m notflyguy.train_chat --chat-types all --force`.

## What the policy is

The fly contributes internal state; it does not contribute language.

| Layer | Role | Where |
| --- | --- | --- |
| Connectome | Fixed recurrent dynamics over 1405 measured neurons / 96,659 measured edges | `notflyguy/connectome.py` |
| Drive | Modeled sensory injection from message features into upstream neurons | `notflyguy/groupchat.py` |
| Readout | Trainable linear model: state + context -> *speak?* and *how long?* | `notflyguy/chat_policy.py` |
| Language | Wording, transport, delivery | Hermes |

Each incoming message advances the circuit once. Only events that already
happened are visible, so the state is causal and no label can leak backwards.

## Where the training data comes from

Directly from the real Hermes transcript database (`~/.hermes/state.db` and
each profile's `state.db`), not from a synthetic corpus. Rows are joined
through `sessions.chat_type` so group conversations can be separated from
direct ones, and sender identity is taken from the per-message
`display_metadata.gateway_input_owner` when present.

Privacy posture, which is enforced in code rather than promised:

- Reads local files only; no network access in the loader.
- Sender and group identifiers are replaced with truncated SHA-256 handles
  before they reach a model, a report, or the dashboard.
- Message text is never written to `report.json`; `ChatEvent.to_public()`
  drops it and reports only a length.

Pseudonymisation is not encryption. The database itself remains the sensitive
artifact.

## Corpus actually available

| Quantity | Value |
| --- | --- |
| Events loaded | 5396 |
| Sessions (all sources) | 158 |
| User messages (decision points, all sources) | 1400 |
| **Group decision points** | **13** |
| **Distinct group conversations** | **3** |
| Distinct participants | 6 |
| Agent messages | 3996 |
| Messages addressed to the fly by name | 15 |

This is the central problem. The project's whole premise is group-chat
behaviour, but only 0.9% of the available decision points come from group
conversations. Those 13 are the only rows that can teach group-specific
behaviour.

The group itself is live, not dormant: the main group session spans 13 days and
gained 8 messages on its most recent day, from 5 distinct participants. Nothing
is being lost to Hermes compaction — every group row is `active=1,
compacted=0`, so the loader sees all of it.

One encouraging detail: **group-only labels are far less skewed than the
all-source mix** — 9 reply / 4 silent (69% reply), against 85% reply across all
sources. The group corpus is genuinely more informative per message, which is
why it is worth collecting separately rather than diluted into the CLI history.

## Are we there yet?

`python -m notflyguy.corpus` (also shown in the dashboard's Training tab) reports
progress against the gates:

```
Group-chat corpus: 13 decision points from 3 session(s), 6 participant(s)
  sessions          3/5
  decision points   13/50
  participants      6/3
  labels            9 reply / 4 silent (69% reply)
  rate              ~6.5/week
NOT READY: group_sessions: 3/5, group_decision_points: 13/50 (~5.7 week(s) at the recent rate)
```

The gate requires distinct **sessions**, not just messages, because evaluation
is leave-sessions-out. A thousand messages from one conversation still yields
one fold and cannot show that a policy transfers to a new group.


## Task

For every user message, predict:

- `respond` — did the agent reply immediately after this message?
- `reply_length` — normalised length of that reply, when there was one.

Base rate for `respond` is **85.3%**, so predicting "always reply" scores 85.3%
and any accuracy at or below that is worthless.

## Method

Leave-sessions-out cross-validation: whole sessions are assigned to folds, so
no message from a held-out conversation is ever seen during training. Two
models are trained per fold — context features only, and context plus
connectome state — because the interesting question is whether the measured
circuit adds anything beyond surface features.

## Results

5 folds, 1400 decision points:

| Feature set | Mean accuracy | Majority baseline | Mean ROC AUC | Pooled accuracy | Pooled Brier |
| --- | --- | --- | --- | --- | --- |
| Context only | 0.8571 | 0.8571 | **0.6519** | 0.8529 | 0.1218 |
| Context + connectome | 0.8594 | 0.8571 | 0.6437 | 0.8557 | 0.1195 |

Reply-length head, against a constant predictor:

| Metric | Value |
| --- | --- |
| Policy MAE | 0.1895 |
| Constant-baseline MAE | **0.1671** |

## What the numbers mean

**The policy does not beat "always reply" on accuracy.** Mean accuracy
(0.8594) is 0.23 percentage points above the majority baseline (0.8571), and
the AUC is above chance (0.64), so it does rank some messages as more
reply-worthy than others — but it is not shown to predict *whether* to speak.

**The connectome does not add predictive value.** Adding the 8 connectome
features moved accuracy by +0.0023, improved **zero** of five folds (four moved
by exactly 0.0000, one by +0.0116), and *reduced* ROC AUC by 0.0081. The
trainer reports `connectome_adds_predictive_value: false`.

**The length head is worse than a constant.** 0.1895 vs 0.1671 MAE. Predicting
the training mean beats the trained ridge head. The dashboard label
"would speak / reply length ≈ N%" therefore carries less information than it
appears to; treat it as a visualisation, not a calibrated estimate.

**The corpus is below the sufficiency threshold.** 13 group decision points
across 3 conversations, against floors of 50 points and 5 conversations. The
trainer therefore emits `sufficient_data: false` and the claim:

> NOT ESTABLISHED: the corpus is below the configured sufficiency thresholds
> (1400 decision points, 13 from actual group chats across 3 session(s);
> thresholds are 200, 50 and 5). The trained artifact is a plumbing check that
> the pipeline runs end to end, not evidence of learned group-chat behaviour.

That verdict is the correct output, not a bug. The pipeline is verified to run
end to end on real transcripts; learned group-chat behaviour is not
demonstrated.

## What would change the answer

The bottleneck is data, not model capacity, and the gates are the honest bar:

1. **Keep using the group.** Every group message Hermes handles is already
   persisted to `state.db`, and the plugin adds nothing to that path, so the
   corpus grows on its own with no extra capture step. Recent activity was
   ~6.5 decision points per week, which puts the 50-point floor roughly 6 weeks
   out at that rate — and the 5-conversation floor depends on how many separate
   group conversations happen, not on volume.
   `python -m notflyguy.corpus` reports both, with a naive linear projection
   that is explicitly labelled as not a forecast.
2. **Re-run with the group-only corpus** to see group-specific behaviour in
   isolation: `python -m notflyguy.train_chat --chat-types group --output data/training/chat-group`.
   With 13 points this will not clear the gate either; it is there to make the
   gap visible.
3. **Only then** consider a richer model. With 13 points across 3
   conversations, a larger model would memorise the sessions and make the
   held-out number worse, not better.

## Reproducing

```bash
python -m notflyguy.chatlog                        # corpus shape, no text
python -m notflyguy.corpus                         # readiness vs the gates
python -m notflyguy.train_chat --chat-types all    # train + cross-validate
```

Then read `verdict` in `data/training/chat-policy-v1/report.json` before
believing any other field.
