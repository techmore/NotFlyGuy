"""Where the derived circuit data lives.

``data/`` is not committed, so every module that loads the measured circuit
needs one agreed place to look. By default that is ``<repo>/data/malecns-v1.0``;
set ``NOTFLYGUY_DATA_ROOT`` to point somewhere else (a shared download, a
synthetic stand-in for tests, a read-only mount). The variable is read at call
time, not import time, so a test or a wrapper script can set it after the
package has been imported.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_ROOT = 'NOTFLYGUY_DATA_ROOT'
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / 'data' / 'malecns-v1.0'

CIRCUIT_FILE = 'motor-circuit.npz'
SUMMARY_FILE = 'motor-circuit.json'
CATALOG_FILE = 'leg-motor-catalog.json'


def data_root(override=None) -> Path:
    """Directory holding ``motor-circuit.npz`` and ``leg-motor-catalog.json``.

    ``override`` wins when given; otherwise ``NOTFLYGUY_DATA_ROOT``; otherwise
    the in-repo default. The directory is not required to exist.
    """
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(ENV_DATA_ROOT, '').strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_DATA_ROOT


def training_root() -> Path:
    """Directory holding trainer output (``data/training/<run>/report.json``)."""
    return REPO_ROOT / 'data' / 'training'
