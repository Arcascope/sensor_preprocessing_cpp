"""Shared fixtures for resampling artifact tests."""

import os
import pytest
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# ── helpers ────────────────────────────────────────────────────────


def load_csv(name: str):
    """Load one of the test CSVs, returning (timestamps_s, x, y, z).

    Handles the three formats found in the data directory:
      - SleepAccel: space-delimited, no header, columns = t x y z (seconds)
      - Weaver/Dreamt: comma-delimited with TIMESTAMP,ACC_X,ACC_Y,ACC_Z header
    """
    path = DATA_DIR / name
    # Sniff first line
    with open(path) as f:
        first = f.readline()

    if first.strip().startswith("TIMESTAMP"):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    else:
        data = np.loadtxt(path)

    t, x, y, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return t, x, y, z


@pytest.fixture(params=["SleepAccel.csv", "Weaver.csv", "Dreamt.csv"])
def real_accel(request):
    """Parametrised fixture yielding (name, t, x, y, z) for each test CSV."""
    name = request.param
    t, x, y, z = load_csv(name)
    return name, t, x, y, z
