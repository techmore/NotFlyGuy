# Wiring control: does the measured connectome earn its keep?

The project calls itself connectome-powered. Until this experiment, that claim
had never been tested: nothing in the repository compared the real measured
wiring against a rewired graph of the same size. A model built on shuffled
weights would have produced the same confident-looking numbers.

This is the missing control, and it mirrors the one in
[ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf), where a randomly
permuted wiring of the same MaleCNS connectome scored train 0.77 / validation
4.27 against 1.02 / 3.84 for the real wiring.

## The control

`notflyguy/wiring_control.py` builds a **degree-preserving rewiring**. Sources
never move; only the targets of existing edges are permuted, so:

**Preserved exactly**

- every neuron's in-degree and out-degree
- the multiset of synapse-count values, and each source's outgoing weight multiset
- the node set, node ordering, motor-neuron and upstream-neuron roles

**Deliberately destroyed**

- which neuron connects to which — and therefore all motifs, clustering, and
  spatial structure, which is precisely what the real wiring might contribute

Swaps that would create a self-loop or duplicate edge are rejected, and the
degree sequence is asserted to be identical after rewiring and before the file
is written. Five independent shuffles are evaluated so the real circuit is
compared against a distribution rather than a single alternative.

## Method

Identical pipeline for real and shuffled circuits: build decision points from
the real Hermes corpus with the circuit as the reservoir, then
leave-sessions-out cross-validation (`train_chat.evaluate_cross_validation`)
with the same folds, seed, and regularisation. Only the wiring differs.

## Result — all sources (1400 decision points, 158 sessions)

| Model | Mean ROC AUC |
| --- | --- |
| Context features only | **0.6519** |
| Real connectome + context | 0.6437 |
| Shuffled connectomes + context (5 seeds) | 0.6449 – 0.6487 (mean 0.6468, sd 0.0013) |

**The measured wiring does not beat its own degree sequence.** The real circuit
scored below *every* one of the five shuffles, with a margin of −0.0049 AUC
against the best shuffled graph. The shuffle spread is tiny (sd 0.0013), so this
is not a case of the real circuit sitting inside a wide noise band — it sits
just below a tight one.

More awkwardly for the premise: **the context-only model beat every
connectome-augmented model**, real or shuffled. On this task, adding the eight
connectome features does not merely fail to help; it slightly hurts.

The verdict, as emitted by the tool:

> NOT DEMONSTRATED: the real wiring did not beat every degree-preserving shuffle
> (real AUC 0.6437 vs shuffled 0.6449-0.6487, mean 0.6468). The measured wiring
> is therefore not shown to contribute anything beyond its degree sequence on
> this task.

## A false positive this tool produced, and the guard added for it

The group-only run (`--chat-types group`) initially reported **SUPPORTED** — the
real wiring beat all shuffles by 0.071 AUC. It was an artifact:

- 13 decision points, 3 sessions → 3 folds
- all three shuffles returned an *identical* AUC (sd 0.0): the metric is
  quantised at that sample size
- context-only AUC was 0.238, worse than chance
- fold-mean AUC 0.8095 against pooled AUC 0.5833 — a gap of 0.226, meaning one
  fold dominated

The comparison now applies the same sufficiency gate as the trainer
(`min_points=200`, `min_sessions=5`) and refuses to make any claim below it,
reporting the ordering "for the record … not evidence either way". A regression
test pins this: a corpus too small to support a claim must return
`sufficient_data: false` even when the AUC ordering favours the real wiring.

That episode is the argument for the gate. With 13 points, leave-sessions-out
AUC will happily order two identical models differently.

## What this does and does not mean

**It means:** on this task — predicting whether the agent replies, using eight
aggregate features read off the circuit — the measured connectome contributes
nothing that the degree sequence does not already provide.

**It does not mean** the connectome is useless in general:

- The readout is eight aggregate features over a 1405-neuron truncated
  motor-centred subgraph. A richer readout over the full state could extract
  more. ngxson's model card is a direct warning here: reading out only from
  "motor output" neurons (their 1,314 descending neurons) **failed to learn at
  all** (train loss 4.7), and only a full-state readout worked. Our motor
  readout uses only 381 motor neurons, and our chat policy uses eight
  aggregates.
- The task is reply/no-reply, dominated by direct-chat history, not group chat.
- The circuit is a truncated 1405-node subgraph of a 49,393-node central brain.

The honest summary is that the project's *architectural* premise (connectome as
a fixed reservoir) is not yet evidenced, and the current readout is a plausible
reason why.

## Reproducing

```bash
python -m notflyguy.wiring_control                       # all sources, 5 shuffles
python -m notflyguy.wiring_control --chat-types group \
    --output data/training/wiring-control-group --force   # group only (underpowered)
```

Reports land in `data/training/wiring-control/report.json`. Read `verdict`
first, then `sufficient_data`.

## Next experiments this suggests

1. **Widen the readout.** Replace the 8 aggregate features with a linear
   projection over all 1405 neurons (fitted, not hand-designed), then re-run
   this control. If real wiring is going to show up anywhere, a readout that can
   actually see the state is where.
2. **Run the same control on the motor task**, where the readout is a full
   59×381 fitted map and the connectome is used in the way the project intends.
3. **Add signed weights.** We currently use non-negative row-normalised
   coupling and disclaim signs entirely, while 89.7% of our circuit's neurons
   have a usable transmitter label. A signed matrix scaled to a target spectral
   radius is both closer to the biology and closer to the reference recipe.
