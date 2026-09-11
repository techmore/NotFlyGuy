from notflyguy.controller import DeterministicController, FlyObservation


def test_deterministic_controller_matches_flybody_action_width() -> None:
    action = DeterministicController().step(FlyObservation(timestep=1))
    assert len(action.motors) == 59
    assert set(action.motors) == {0.0}

