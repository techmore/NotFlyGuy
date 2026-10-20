"""Persona: shared character text, guidance, and drift against the Hermes copy."""
import importlib.util
from pathlib import Path

from notflyguy.persona import ALIASES, IDENTITY, PERSONA, style_guidance, system_prompt, technical_question

PLUGIN_PATH = Path(__file__).parents[1] / 'integrations/hermes/notflyguy/__init__.py'


def _load_plugin():
    spec = importlib.util.spec_from_file_location('fly_plugin_persona', PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_persona_has_not_drifted():
    """The plugin carries its own copy; this is what keeps the two identical."""
    plugin = _load_plugin()
    assert plugin.PERSONA == PERSONA
    assert tuple(plugin.ALIASES) == tuple(ALIASES)
    assert plugin.IDENTITY == IDENTITY
    assert plugin.TECHNICAL.pattern == __import__('notflyguy.persona', fromlist=['x']).TECHNICAL.pattern


def test_technical_question_detection():
    assert technical_question('how many neurons do you have?')
    assert technical_question('what is the connectome?')
    assert not technical_question('hey, how is it going?')


def test_system_prompt_is_the_persona_without_a_decision():
    assert system_prompt(None, 'direct') == PERSONA
    assert system_prompt({}, 'direct') == PERSONA
    assert system_prompt({'policy_loaded': False}, 'direct') == PERSONA


def test_group_prompt_explains_speaker_attribution():
    """Regression: without this the model conflated participants' names."""
    prompt = system_prompt(None, 'group')
    assert prompt.startswith(PERSONA)
    assert 'group chat with several people' in prompt
    assert 'never attribute' in prompt
    assert 'group chat' not in system_prompt(None, 'direct')


def test_speaker_note_names_the_current_speaker():
    from notflyguy.persona import speaker_note
    assert speaker_note('') == ''
    assert speaker_note(None) == ''
    assert speaker_note('   ') == ''
    note = speaker_note('Carol')
    assert 'from Carol' in note
    assert 'Address Carol directly' in note
    assert "Do not assume another participant's name" in note


def test_speaker_note_reaches_the_group_prompt_only():
    prompt = system_prompt(None, 'group', speaker='Carol')
    assert 'from Carol' in prompt
    assert 'from Carol' not in system_prompt(None, 'direct', speaker='Carol')
    assert 'from Carol' not in system_prompt(None, 'group')


def test_group_prompt_orders_speaker_note_before_policy_guidance():
    decision = {'policy_loaded': True, 'respond_probability': 0.8,
                'predicted_reply_length': 0.5, 'connectome': {'rms': 0.2}}
    prompt = system_prompt(decision, 'group', speaker='Bob')
    assert prompt.index('from Bob') < prompt.index('learned readout')


def test_guidance_is_empty_without_a_policy():
    assert style_guidance(None, 'group') == ''
    assert style_guidance({'policy_loaded': False}, 'group') == ''
    assert style_guidance({'policy_loaded': True}, 'group') == ''


def test_guidance_still_requires_an_answer():
    decision = {'policy_loaded': True, 'respond_probability': 0.05,
                'predicted_reply_length': 0.2, 'connectome': {'rms': 0.3}}
    guidance = style_guidance(decision, 'group')
    assert 'must still answer' in guidance
    assert 'advisory' in guidance
    assert 'do not reply' not in guidance.lower()


def test_guidance_scales_verbosity():
    short = {'policy_loaded': True, 'respond_probability': 0.9,
             'predicted_reply_length': 0.1, 'connectome': {'rms': 0.0}}
    long = dict(short, predicted_reply_length=0.95)
    assert 'sentence or two' in style_guidance(short, 'group')
    assert 'few sentences' in style_guidance(long, 'group')


def test_system_prompt_appends_guidance():
    decision = {'policy_loaded': True, 'respond_probability': 0.8,
                'predicted_reply_length': 0.5, 'connectome': {'rms': 0.2}}
    prompt = system_prompt(decision, 'group')
    assert prompt.startswith(PERSONA)
    assert 'learned readout would speak' in prompt
    assert prompt.index('group chat with several people') < prompt.index('learned readout')


def test_guidance_handles_a_missing_connectome_block():
    decision = {'policy_loaded': True, 'respond_probability': 0.8}
    assert style_guidance(decision, 'group')
