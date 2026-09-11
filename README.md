# NotFlyGuy

NotFlyGuy is an experimental connectome-powered fly agent: a recurrent
controller built on the released **MaleCNS v1.0** connectome, the
**FlyBody/MuJoCo** simulated body, and an observable dashboard. This revision
adds the piece the project was originally aiming at — a **trainable group-chat
policy** wired through the measured circuit.

## How the connectome connects to group chat

The honest architecture, because it matters:

- **Hermes** does language, transport, and reply delivery. It is not the fly.
- **The connectome** is a fixed recurrent dynamical system over 1405 measured
  neurons and 96,659 measured edges. Incoming messages are injected as modeled
  sensory drive; its activity is the fly's internal state.
- **The trained readout** maps that state plus surface context onto two
  decisions: *would the fly speak now*, and *how long should the reply be*.
- **The policy does not write words.** Wording stays with the conversational
  model. Nothing here is a language model, and the circuit is a truncated
  motor-centred subgraph, not a brain.

See [docs/groupchat-policy.md](docs/groupchat-policy.md) for the measured
results and their limits, and [docs/wiring-control.md](docs/wiring-control.md)
for the control that tests whether the measured wiring beats a rewired graph of
the same size. It currently does not.

## Setup

`data/` is not committed, so a fresh clone must reproduce the circuit first.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,connectome,body]'
python scripts/bootstrap_data.py     # download + extract the measured circuit
```

`bootstrap_data.py` runs the three derivation steps in order: fetch and
checksum the published MaleCNS feather tables, catalog the annotated leg motor
neurons, then induce the motor-centred subgraph.

## Run the dashboard

```bash
python -m notflyguy.server          # http://127.0.0.1:8765
python -m notflyguy.server --no-body   # chat circuit + policy only, no MuJoCo
python -m notflyguy.server --no-replies   # monitor only, generate nothing
```

The dashboard is a working chat interface. Type into **Group chat** and the fly
answers: the message drives the measured circuit, the trained readout produces
the decision you see in the chip, and the words come from a local
OpenAI-compatible language model (`--llm-url`, default `http://127.0.0.1:8083`;
set `NOTFLYGUY_LLM_URL` / `NOTFLYGUY_LLM_MODEL` to change it). Without a
reachable model the dashboard still runs and degrades to monitoring.

The readout is displayed, **not enforced**. It does not beat its baselines, so
by default it is not allowed to suppress a reply; `--policy-gate` enables that
for experiments and will make the fly ignore people.

Three views:

- **Group chat** — talk to the fly, watch the circuit advance per message, and
  see the decision chip and the reply. Real Signal traffic appears here as
  events carrying the fly's decision; Hermes owns those replies, so their text
  is deliberately not sent to the browser.
- **Research** — the live FlyBody frame (served as `/frame.jpg`, not inlined
  into the event stream) plus controller telemetry.
- **Training** — corpus readiness against the gates, and the raw training
  reports including verdicts that say `NOT ESTABLISHED`. Those are real results,
  not missing data.

Known limitation: on the local 2B 4-bit model, speaker attribution across a
group conversation is unreliable — it will muddle who said what. That is a model
capability limit; see `notflyguy/persona.py` for the measurements.

## Train the group-chat policy

The trainer reads the **real Hermes transcript database** directly, not a
synthetic corpus. Sender and group identifiers are pseudonymised on load and
message text is never written into a report.

```bash
python -m notflyguy.chatlog                       # inspect the corpus shape
python -m notflyguy.corpus                        # are we ready to train yet?
python -m notflyguy.train_chat --chat-types all   # train + cross-validate
python -m notflyguy.train_chat --chat-types group --output data/training/chat-group
```

Evaluation uses leave-sessions-out folds, compares against the majority-class
baseline, and trains a second model without connectome features to test whether
the circuit adds anything at all. Read the `verdict` field before believing
anything. The gate requires distinct group **conversations**, not just messages,
because leave-sessions-out cannot generalize from one conversation however long
it is. `notflyguy-corpus` and the dashboard's Training tab report progress
against those gates.

## Train the motor controller

```bash
python -m notflyguy.benchmark --steps 200 --seeds 2   # baselines
python -m notflyguy.train --steps 200                 # readout search (CEM)
```

The action spec is **not** `[0, 1]` for every actuator: leg actuators span their
own ranges (`coxa_T1_left` is `[-3, 3]`). The readout maps onto each actuator's
declared range. The all-zeros command is a strong do-nothing local optimum at
`~0.166` mean reward; the spec-maximum posture scores `~0.204`. No run has yet
completed a held-out evaluation that beats the zero baseline by a real margin.
See [docs/learning-baseline.md](docs/learning-baseline.md).

## Tests

```bash
python -m pytest -q
```

No test requires FlyBody, MuJoCo, or a live server; the simulator is faked and
the corpus is synthetic. The circuit tests skip themselves when `data/` has not
been bootstrapped.

## Development notes

Targets Python 3.10–3.12 because FlyBody's documented environment uses 3.10.
Python 3.14 is intentionally not used for the simulator environment.

The native [Hermes integration](integrations/hermes/README.md) registers `/fly`
commands, feeds the circuit, and injects the policy reading as advisory style
guidance. Hermes cannot suppress a send from a plugin hook, so the policy
advises rather than gates — see that README for the exact contract.
