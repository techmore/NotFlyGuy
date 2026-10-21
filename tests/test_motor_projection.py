"""Motor readout: action range mapping, leg-local mask, parameter plumbing."""
import numpy as np
import pytest

from notflyguy.motor_projection import ACTION_MAX, ACTION_MIN, ACTION_NEUTRAL, LegMotorReadout, to_action

ACTUATORS = (
    'walker/head_abduct', 'walker/abdomen',
    'walker/coxa_T1_left', 'walker/femur_T1_left', 'walker/tibia_T1_left',
    'walker/coxa_T1_right', 'walker/femur_T1_right',
    'walker/coxa_T2_left', 'walker/tibia_T3_right',
    'walker/adhere_T1_left', 'walker/coxa_T0_left',
)
NEURONS = (
    {'bodyId': 1, 'somaNeuromere': 'T1', 'somaSide': 'L'},
    {'bodyId': 2, 'somaNeuromere': 'T1', 'somaSide': 'R'},
    {'bodyId': 3, 'somaNeuromere': 'T2', 'somaSide': 'L'},
    {'bodyId': 4, 'somaNeuromere': 'T3', 'somaSide': 'R'},
    {'bodyId': 5, 'somaNeuromere': 'T2', 'somaSide': 'R'},
)


@pytest.fixture
def readout():
    return LegMotorReadout(NEURONS, ACTUATORS, seed=0)


def test_to_action_stays_inside_the_flybody_range():
    values = to_action(np.array([-100.0, -1.0, 0.0, 1.0, 100.0]))
    assert values.min() >= ACTION_MIN
    assert values.max() <= ACTION_MAX
    assert to_action(np.array([0.0]))[0] == pytest.approx(ACTION_NEUTRAL)


def test_untrained_readout_commands_the_neutral_posture(readout):
    actions = readout(np.zeros(len(NEURONS)))
    assert actions == pytest.approx(np.full(len(ACTUATORS), ACTION_NEUTRAL))


def test_mask_is_leg_local_and_skips_non_leg_actuators(readout):
    names = list(readout.actuator_names)
    for row, name in enumerate(names):
        if 'adhere' in name or 'head' in name or 'abdomen' in name or 'T0' in name:
            assert not readout.mask[row].any(), name
            continue
        segment = name.split('_')[-2]
        side = name.split('_')[-1]
        assert readout.mask[row].any(), name
        for column in np.flatnonzero(readout.mask[row]):
            assert NEURONS[column]['somaNeuromere'] == segment
            assert {'L': 'left', 'R': 'right'}[NEURONS[column]['somaSide']] == side


def test_parameters_round_trip_through_the_mask(readout):
    flat = readout.parameters()
    assert flat.shape == (int(readout.mask.sum()) + len(ACTUATORS),)
    readout.bias = np.arange(len(ACTUATORS), dtype=float)
    round_tripped = readout.parameters()
    fresh = LegMotorReadout(NEURONS, ACTUATORS, seed=1)
    fresh.set_parameters(round_tripped)
    assert np.allclose(fresh.weights, readout.weights)
    assert np.allclose(fresh.bias, readout.bias)
    assert np.allclose(fresh(np.ones(len(NEURONS))), readout(np.ones(len(NEURONS))))


def test_masked_weights_stay_zero_outside_the_mask(readout):
    flat = readout.parameters()
    flat[:] = 1.0
    readout.set_parameters(flat)
    assert np.allclose(readout.weights[~readout.mask], 0.0)
    assert np.allclose(readout.weights[readout.mask], 1.0)


def test_set_parameters_rejects_wrong_shapes_and_non_finite(readout):
    good = readout.parameters()
    with pytest.raises(ValueError):
        readout.set_parameters(good[:-1])
    with pytest.raises(ValueError):
        readout.set_parameters(np.full(len(good), np.nan))


def test_bias_gives_control_authority_without_activity(readout):
    """The old readout could only output ~0 with zero activity; bias fixes that."""
    readout.bias = np.full(len(ACTUATORS), -3.0)
    low = readout(np.zeros(len(NEURONS)))
    readout.bias = np.full(len(ACTUATORS), 3.0)
    high = readout(np.zeros(len(NEURONS)))
    assert low.mean() < ACTION_NEUTRAL < high.mean()
    assert high.mean() - low.mean() > 0.5


def test_call_validates_activity(readout):
    with pytest.raises(ValueError):
        readout(np.zeros(len(NEURONS) + 1))
    with pytest.raises(ValueError):
        readout(np.full(len(NEURONS), np.nan))


# --- Per-actuator ranges ---------------------------------------------------
# The live FlyBody spec is not [0, 1] for every actuator (coxa_T1_left is
# [-3, 3], head is [0, 1]). A fixed (0, 1) map makes most postures unreachable.
MIXED_LIMITS = np.array([
    [0.0, 1.0],       # head
    [0.0, 1.0],       # abdomen
    [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0],
    [-3.0, 3.0], [-3.0, 3.0],
    [-3.0, 3.0], [-3.0, 3.0],
    [0.0, 1.0],       # adhere
    [0.0, 1.0],       # unmapped actuator
])


def test_limits_normalisation_accepts_a_shared_range():
    from notflyguy.motor_projection import normalize_limits
    assert normalize_limits(None, 3).tolist() == [[0.0, 1.0]] * 3
    assert normalize_limits([-2.0, 5.0], 2).tolist() == [[-2.0, 5.0]] * 2
    with pytest.raises(ValueError):
        normalize_limits([1.0, 0.0], 2)
    with pytest.raises(ValueError):
        normalize_limits(np.zeros((3, 2)), 2)
    with pytest.raises(ValueError):
        normalize_limits([[0.0, np.nan]], 1)


def test_limits_from_spec_reads_the_action_spec():
    from notflyguy.motor_projection import limits_from_spec
    spec = type('Spec', (), {'minimum': np.array([0.0, -3.0]),
                             'maximum': np.array([1.0, 3.0])})()
    assert limits_from_spec(spec).tolist() == [[0.0, 1.0], [-3.0, 3.0]]


def test_readout_respects_per_actuator_ranges():
    readout = LegMotorReadout(NEURONS, ACTUATORS, seed=0, limits=MIXED_LIMITS)
    # Zero drive must land on each actuator's own midpoint, not a global 0.5.
    neutral = readout(np.zeros(len(NEURONS)))
    assert neutral == pytest.approx(MIXED_LIMITS.mean(axis=1))
    assert neutral[2] == pytest.approx(0.0)   # [-3, 3] -> centre 0
    assert neutral[0] == pytest.approx(0.5)   # [0, 1]  -> centre 0.5

    # Extreme drive stays inside the declared range on both sides.
    readout.bias = np.full(len(ACTUATORS), 50.0)
    high = readout(np.zeros(len(NEURONS)))
    readout.bias = np.full(len(ACTUATORS), -50.0)
    low = readout(np.zeros(len(NEURONS)))
    assert (high <= MIXED_LIMITS[:, 1] + 1e-9).all()
    assert (low >= MIXED_LIMITS[:, 0] - 1e-9).all()
    assert high[2] > 2.9 and low[2] < -2.9


def test_to_action_honours_explicit_limits():
    values = to_action(np.array([-50.0, 0.0, 50.0]), limits=np.array([[-3.0, 3.0]] * 3))
    assert values[0] < -2.9
    assert values[1] == pytest.approx(0.0)
    assert values[2] > 2.9


def test_parameter_vector_is_unchanged_by_limits(readout):
    """Limits must not change the trainable parameter layout."""
    limited = LegMotorReadout(NEURONS, ACTUATORS, seed=0, limits=MIXED_LIMITS)
    assert readout.parameters().shape == limited.parameters().shape
    limited.set_parameters(readout.parameters())
    assert np.allclose(limited.weights, readout.weights)
