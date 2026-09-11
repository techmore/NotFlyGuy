# NotFlyGuy

NotFlyGuy is an experimental connectome-powered fly agent: a MaleCNS-inspired
neural controller, the FlyBody/MuJoCo simulated body, and an observable agent
layer for chat, training, and scheduled goals.

## Build stages

1. **Foundation** — repository structure, contracts, configuration, and a
   deterministic controller demo.
2. **Body** — install FlyBody and drive a MuJoCo fly through the controller
   interface; render frames and collect telemetry.
3. **Brain** — load a small MaleCNS subgraph, implement sparse recurrent
   dynamics, and connect sensory/motor adapters.
4. **Learning** — add rewards, checkpoints, evaluation episodes, and a first
   trainable locomotion task.
5. **Presence** — live dashboard, replay viewer, and a private Tailscale
   endpoint.
6. **Agent** — group-chat adapter, LLM coordinator, tool permissions, and
   scheduled goals with human approval for consequential actions.

## Current status

Stage 0 is complete. The controller contract is intentionally independent of
FlyBody and the eventual neural implementation, so we can test the system with
a deterministic placeholder before taking on the heavier simulator setup.

## Development

The first prototype targets Python 3.10–3.12 because FlyBody's documented
environment uses Python 3.10. Python 3.14 is intentionally not used for the
initial simulator environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m notflyguy.demo
```

