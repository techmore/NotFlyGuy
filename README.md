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

## Install

Python 3.10–3.12 (FlyBody's documented environment is 3.10; the core package
needs only numpy and scipy).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'              # core + pytest + ruff
```

Optional extras:

| extra        | adds                                   | needed for                                   |
|--------------|----------------------------------------|----------------------------------------------|
| `connectome` | `pyarrow`                              | reading the released MaleCNS feather tables  |
| `body`       | FlyBody (pinned git commit) + Pillow   | the MuJoCo body and the dashboard's frames   |
| `all`        | everything above                       |                                              |

### Get the measured circuit

`data/` is not committed. Reproduce the circuit from the published MaleCNS
v1.0 release (a large download) once per clone:

```bash
pip install -e '.[connectome]'
python scripts/bootstrap_data.py     # fetch + checksum, catalog leg motor neurons, extract subgraph
```

`bootstrap_data.py` runs the three derivation steps in order and writes
`data/malecns-v1.0/{motor-circuit.npz,motor-circuit.json,leg-motor-catalog.json}`.

Every loader resolves that directory through `notflyguy.paths.data_root()`:
set **`NOTFLYGUY_DATA_ROOT`** to use a circuit stored elsewhere (a shared
download, a read-only mount, or the synthetic stand-in below).

### Run without the download

A random circuit with the same file layout lets you exercise the dashboard,
the trainer plumbing, and the tests without fetching anything. It is **not
measured biology** and says so in its `motor-circuit.json` (`"synthetic": true`).

```bash
python -m notflyguy.synthetic --root /path/to/fake-circuit
NOTFLYGUY_DATA_ROOT=/path/to/fake-circuit python -m notflyguy.server --no-body
```

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

## Tests and lint

```bash
python -m pytest -q      # ~25 s
ruff check .             # lint + import order; `ruff check . --fix` applies the safe fixes
```

No test requires FlyBody, MuJoCo, a live server, or the MaleCNS download: the
simulator is faked, the Hermes corpus is synthetic, and when `data/` has not
been bootstrapped the test session writes a synthetic circuit once and points
`NOTFLYGUY_DATA_ROOT` at it (`tests/conftest.py`). If the measured circuit is
present it is used instead. Tests assert contracts -- shapes, determinism,
stability bounds, labels -- never biological results, so they are valid on
either.

CI (`.github/workflows/ci.yml`) runs ruff and the suite on Python 3.10, 3.11
and 3.12 on every push and pull request, then boots the dashboard against a
synthetic circuit and checks `/api/state`.

## Environment variables

| variable                | default                              | purpose                                             |
|-------------------------|--------------------------------------|-----------------------------------------------------|
| `NOTFLYGUY_DATA_ROOT`   | `data/malecns-v1.0`                  | directory holding the circuit and leg catalog        |
| `NOTFLYGUY_LLM_URL`     | `http://127.0.0.1:8083`              | OpenAI-compatible endpoint for dashboard replies     |
| `NOTFLYGUY_LLM_MODEL`   | first model the endpoint lists       | model name sent with each completion                 |
| `NOTFLYGUY_HERMES_DB`   | `~/.hermes/**/state.db`              | colon-separated Hermes databases for the trainer     |
| `NOTFLYGUY_GATEWAY_LOG` | `~/.hermes/profiles/not-rocky/logs/agent.log` | gateway log the dashboard watches for activity |
| `NOTFLYGUY_ID_SALT`     | `notflyguy/v1`                       | pseudonymisation salt for exported identifiers (read at import) |
| `NOTFLYGUY_BASE`        | `http://127.0.0.1:8765`              | habitat URL the Hermes plugin talks to               |
| `NOTFLYGUY_DR_AUTOSTART`| unset                                | `1` starts the DragonRealms pilot with the server (needs `NOTFLYGUY_DR_URL/USER/PASSWORD`) |

## Development notes

Targets Python 3.10–3.12 because FlyBody's documented environment uses 3.10.
Python 3.14 is intentionally not used for the simulator environment.
`python -m notflyguy` is a shortcut for `python -m notflyguy.server`.

The native [Hermes integration](integrations/hermes/README.md) registers `/fly`
commands, feeds the circuit, and injects the policy reading as advisory style
guidance. Hermes cannot suppress a send from a plugin hook, so the policy
advises rather than gates — see that README for the exact contract.
