from __future__ import annotations

import numpy as np


class MANUAL:
    DEFAULT_HEIGHT = 1.8
    MIN_HEIGHT = 0.45
    MAX_HEIGHT = 5.5

    YAW_RATE_CMD = np.deg2rad(55.0)
    CLIMB_RATE_CMD = 0.85
    FORWARD_CMD = 0.18
    BACKWARD_CMD = -0.12
