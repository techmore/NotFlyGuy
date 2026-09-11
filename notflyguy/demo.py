"""Run the Stage 0 controller contract without simulator dependencies."""

from .controller import DeterministicController, FlyObservation


def main() -> None:
    controller = DeterministicController()
    action = controller.step(FlyObservation(timestep=0))
    print(f"NotFlyGuy Stage 0 online: {len(action.motors)} motor channels")


if __name__ == "__main__":
    main()

