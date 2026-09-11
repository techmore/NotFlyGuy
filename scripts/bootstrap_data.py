"""Fetch and extract the measured circuit, in the order the pipeline requires.

``data/`` is intentionally not committed, so a fresh clone has no circuit and
the connectome-backed modules cannot start. Running this once reproduces every
derived artifact from the published MaleCNS v1.0 release.

    python scripts/bootstrap_data.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = (
    'fetch_connectome.py',      # download + checksum the released feather tables
    'catalog_leg_neurons.py',   # select annotated leg motor neurons
    'extract_motor_circuit.py',  # induce and save the motor-centred subgraph
)


def main() -> int:
    for step in STEPS:
        script = HERE / step
        print(f'--- {step}', flush=True)
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f'FAILED: {step} exited {result.returncode}', file=sys.stderr)
            return result.returncode
    print('Circuit data ready under data/malecns-v1.0/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
