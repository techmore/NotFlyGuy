"""The fly's conversational character, shared by every surface that speaks.

Canonical copy. The Hermes plugin carries its own copy because it is linked
into a profile's plugin directory and cannot rely on this package being
importable; ``tests/test_persona.py`` asserts the two stay identical so the
duplication cannot silently drift.
"""
from __future__ import annotations

import re

IDENTITY = 'NotFlyGuy'
ALIASES = ('notflyguy', 'not-rocky', 'not rocky')

PERSONA = (
    'For this conversation, speak as NotFlyGuy, the little fly at a keyboard. '
    'Be a friendly, curious, lightly mischievous conversational companion, not a diagnostic endpoint. '
    'Respond to what the person actually said and follow the conversation. '
    'For greetings or "how is it going", give a short natural social reply, perhaps a small fly joke; '
    'do not report running state, steps, neurons, reward, controller details, or viewer URLs. '
    'Do not call habitat tools or load skills for small talk. Do not recycle a fixed catchphrase. '
    'Do not introduce yourself as Rocky or Not-Rocky; that is the transport account, not your character. '
    'Technical details belong only in answers to explicit technical questions. '
    'Do not pretend the biological connectome creates your language or that animated typing is learned behavior. '
    'Do not claim actions, perception, memories, or training outcomes you do not have evidence for. '
    'Earlier telemetry-heavy greetings were a configuration mistake, not the desired conversation style.'
)

TECHNICAL = re.compile(
    r'\b(neurons?|connectome|telemetry|simulation|controller|reward|rms|viewer|habitat|status)\b', re.I)

# Observed failure without this: asked "what is my name?" by Carol after Sean had
# introduced himself, the model answered "Your name is Carol!" -- it had the
# attribution in the history but conflated speakers. Group turns are labelled,
# so the fix is to say plainly that the labels matter.
GROUP_GUIDANCE = (
    'You are in a group chat with several people, not a private conversation. '
    "Every incoming message is prefixed with its speaker's name. Reply to whoever "
    'just spoke, and keep track of which participant said what: never attribute one '
    "person's name or words to another. Address people by name when it helps. Keep "
    'group replies short unless someone asks for detail.'
)


def style_guidance(decision: dict | None, chat_kind: str = 'group') -> str:
    """Turn a policy reading into advisory wording for the language model."""
    if not decision or not decision.get('policy_loaded'):
        return ''
    probability = decision.get('respond_probability')
    length = decision.get('predicted_reply_length')
    if probability is None:
        return ''
    where = 'group chat' if (chat_kind or 'group') == 'group' else 'conversation'
    connectome = decision.get('connectome') or {}
    rms = connectome.get('rms', 0.0) or 0.0
    lines = [
        f'The fly control policy read this {where} message with internal state from '
        f'the measured connectome (circuit activity RMS {rms:.3f}).',
    ]
    if probability < 0.5:
        lines.append(
            f'Its learned readout would hold back here (p={probability:.2f}). This is advisory '
            'only: you must still answer, but prefer a light, brief contribution over a long reply.'
        )
    else:
        lines.append(f'Its learned readout would speak here (p={probability:.2f}).')
    if length is not None:
        if length > 0.7:
            lines.append('It expects a fuller reply; a few sentences are appropriate.')
        elif length < 0.4:
            lines.append('It expects a short reply; keep it to a sentence or two.')
    lines.append('Do not mention the policy numbers or the connectome unless explicitly asked.')
    return '\n'.join(lines)


def speaker_note(speaker: str | None) -> str:
    """State plainly who is talking, and warn against borrowing another's name.

    Honest status: this is a small, free improvement, not a fix. Measured against
    the local 2B 4-bit model, a group "what is my name?" turn still produced
    "Carol, you're Carol!" and "Carol, you're the one who just said your name!"
    across temperatures 0.0-0.7. The model reliably answered correctly only when
    the prompt was handed the facts ("Carol has not introduced herself; only Bob
    claimed the name Sean"), i.e. when the answer was supplied.

    Speaker attribution in a multi-party conversation is therefore a model
    capability limit here, not something further prompt wording resolves. Use a
    stronger model for group chats, or accept that names get muddled.
    """
    speaker = (speaker or '').strip()
    if not speaker:
        return ''
    return (f'The message you are answering is from {speaker}. Address {speaker} directly. '
            f"Do not assume another participant's name or words belong to {speaker}.")


def system_prompt(decision: dict | None = None, chat_kind: str = 'group',
                  speaker: str | None = None) -> str:
    """Persona, group awareness, current speaker, and advisory policy guidance."""
    parts = [PERSONA]
    if (chat_kind or 'group') == 'group':
        parts.append(GROUP_GUIDANCE)
        note = speaker_note(speaker)
        if note:
            parts.append(note)
    guidance = style_guidance(decision, chat_kind)
    if guidance:
        parts.append(guidance)
    return '\n'.join(parts)


def technical_question(message: str) -> bool:
    return bool(TECHNICAL.search(str(message)))
