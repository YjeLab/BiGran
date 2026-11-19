"""Spectrum grid constants shared by the model and label space."""
from __future__ import annotations

import numpy as np

N_POINTS = 1800
WN_START = 500.0
WN_STEP = 2.0
N_COARSE, N_FINE = 17, 72


def wavenumber_grid(n_points: int = N_POINTS, start: float = WN_START, step: float = WN_STEP) -> np.ndarray:
    return start + step * np.arange(n_points, dtype=np.float64)