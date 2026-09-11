# Hermes / Signal connection

The native Hermes plugin at `integrations/hermes/notflyguy/` provides:

- `/fly status`, `/fly pause`, `/fly resume`, `/fly reset`, `/fly <goal>`;
- the `notflyguy_habitat` tool;
- a `pre_llm_call` hook that injects the fly persona, the group-chat policy
  reading, and (for explicit technical questions) fresh habitat telemetry;
- a `post_llm_call` hook that feeds the agent's own reply back to the circuit
  and updates the local dashboard.

Hermes owns authentication, incoming Signal messages, and reply delivery. The
plugin calls the habitat at `http://127.0.0.1:8765`; no Signal credentials or
group identifiers belong in this repository.

## What the policy can and cannot do

This is a hard constraint of the Hermes plugin API, verified in
`~/.hermes/hermes-agent/website/docs/user-guide/features/hooks.md`:

| Hook | Category | Return behaviour |
| --- | --- | --- |
| `pre_llm_call` | Directive/control | Valid string / `{"context": ...}` returns are **joined and injected** into the user message. |
| `post_llm_call` | Observer | Return is **ignored**. |
| `transform_llm_output` | Transform | First **non-empty** string replaces the response. |

There is no plugin surface that suppresses a send. The trained policy therefore
**advises** rather than **gates**: it tells the model whether its readout would
speak, and nudges verbosity, but the model still answers. The injected text says
so explicitly (`advisory only: you must still answer`), and a test asserts the
guidance never tells the model to stay silent.

Do not describe this as the fly deciding whether to reply. Making that true
would need a different mechanism, such as Hermes group-trigger configuration,
not a `pre_llm_call` return value.

The plugin looks up `chat_type` for the session in the Hermes SQLite database so
the policy knows whether it is in a group; when the lookup fails it degrades to
the group default rather than raising.

## Training data

Every Signal turn Hermes records is already written to its `state.db`. The
trainer reads that database directly, so the group-chat corpus grows on its own
and no extra capture path is required. The plugin writes only policy
*predictions* to `data/training/chat-policy-v1/shadow.jsonl`; message text is
never written there.

The current corpus is far too small for group-specific learning — see
[docs/groupchat-policy.md](../../docs/groupchat-policy.md).

## Install

Link `notflyguy/` into the active Hermes profile's `plugins/` directory, run
`hermes --profile PROFILE plugins enable notflyguy`, and restart its gateway.
Built-in tool override permission is not required.

Local setup: enabled in the default and `not-rocky` profiles. The existing
gateway routes Signal through `not-rocky`, whose Signal adapter was enabled.

## Verification status

Carried forward from earlier work, and still accurate:

- The plugin passed Hermes Plugin Doctor; the running gateway reported Signal
  connected.
- A direct plugin call retrieved a current FlyBody image and timestep.
- `/fly pause`, `/fly resume`, and `/fly reset` queue operations on the MuJoCo
  main thread and were verified against the live simulator.
- The profile's real Hermes command registry resolved `/fly status`.
- Two fresh-session CLI tests (one toolset-restricted, one with the normal tool
  configuration) executed the live habitat tool and answered correctly. The
  second turn took about 48 seconds on the local model.
- The profile uses its existing local MLX model on port 8083; the MLX process
  and habitat run as launchd services `ai.hermes.mlx-e2b` and
  `com.techmore.notflyguy`.

Still **not** verified, and not claimed:

- Actual incoming-message to delivered-reply behaviour in the existing Signal
  group. Neither CLI test is evidence of end-to-end Signal success.
- Whether the injected policy guidance improves real replies. The policy does
  not beat its baselines on held-out data, so there is no basis to expect it to.
- Remote browser access. The viewer is localhost-only; Tailscale is not
  configured.
- Goals are recorded but do not produce trained motor behaviour.
