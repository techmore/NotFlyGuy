---
name: notflyguy-habitat
description: Read live NotFlyGuy habitat status through its registered Hermes tool. Also resolves legacy requests that confuse the tool name with a skill.
---

# NotFlyGuy habitat

The integration is installed. This skill provides instructions; live data comes
from the registered `notflyguy_habitat` tool.

## When to call the tool

Call `notflyguy_habitat` with `{"command":"status"}`, or use `notflyguy_habitat`
as the target of a tool dispatcher with the same argument object, **only when
someone explicitly asks about the fly's state** — running or paused, step count,
neuron or edge counts, reward, or the viewer link.

Do **not** call it for greetings, small talk, or "how's it going". For those,
reply as the fly in character. Reporting step counts and neuron counts at a
greeting is a configuration mistake, not the intended behaviour. The persona
instructions injected for the turn take precedence over any habit of checking
status first.

Only use `pause`, `resume`, or `reset` when explicitly requested.

## Reporting

When you do report status, give the requested fields plainly and include the
viewer link. Do not pad a technical answer with unrelated telemetry.

If the tool is absent or fails, say the live check failed and suggest
`/fly status` in Signal. Do not invent telemetry, run installation commands, or
claim that the fly generates language.

Do not say the skill failed to load: reaching these instructions means loading
succeeded.

## What the project actually is

The web desk is an animated chat interface; its simulator replies are scripted
and are never sent to Signal.

The separate research view and the group-chat policy use an experimental
connectome-derived controller over a truncated, motor-centred subgraph of the
MaleCNS v1.0 connectome. The trained readout decides whether the fly *would*
speak and roughly how long a reply should be. It writes no words, it is not a
language model, and it does not beat its own baselines on held-out data. Its
readings are advisory only.
