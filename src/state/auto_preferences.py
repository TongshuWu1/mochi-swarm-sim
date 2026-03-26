from __future__ import annotations

import numpy as np


class AUTO:
    TARGET_SEQUENCE = [
        "red",
        "orange",
        "yellow",
        "lime",
        "green",
        "cyan",
        "blue",
        "purple",
        "magenta",
    ]
    TARGET_HEIGHT_HINTS = [2.4, 2.9, 3.4, 2.8, 3.6, 2.6, 3.2, 2.5, 3.0]

    MIN_HEIGHT = 0.7
    MAX_HEIGHT = 4.8
    DEFAULT_HEIGHT = 1.8

    LOST_TIMEOUT = 0.15
    MIN_TRACK_CONFIDENCE = 0.20


class AUTO_TRACK:
    YAW_GAIN = 0.22
    MAX_YAW_STEP = np.deg2rad(7.5)
    HEIGHT_GAIN = 0.58
    MAX_HEIGHT_STEP = 0.10

    FORWARD_THRUST_FAR = 0.08
    FORWARD_THRUST_MID = 0.13
    FORWARD_THRUST_NEAR = 0.15

    OFFSET_ALPHA = 0.30
    SIZE_ALPHA = 0.25


class AUTO_CHARGE:
    ALIGNMENT_RADIUS = 0.3
    TIGHT_CENTER_RADIUS = 0.070
    CENTER_RADIUS = ALIGNMENT_RADIUS

    AREA_THRESHOLD = 900.0
    WIDTH_FRAC_THRESHOLD = 0.25
    HEIGHT_FRAC_THRESHOLD = 0.25
    CONFIRM_FRAMES = 2
    CONFIRM_TIME = 0.10

    THRUST = 0.30
    DURATION = 5.0
    RAMP_TIME = 0.30
    RAMP_START_THRUST = 0.18

    EARLY_ADVANCE_MIN_TIME = 0.35
    EARLY_ADVANCE_LOST_TIME = 0.10


class AUTO_SEARCH:
    # Search now uses nearly constant forward speed and a large initial turn rate
    # that decays over time. With approximately constant forward motion, that gives
    # an outward spiral in practice instead of the earlier tight curving search.
    SPIRAL_THRUST = 0.024

    SPIRAL_INITIAL_YAW_RATE = np.deg2rad(140.0)
    SPIRAL_FINAL_YAW_RATE = np.deg2rad(24.0)
    SPIRAL_RATE_DECAY_TIME = 8.0

    MAX_YAW_LEAD_START = np.deg2rad(18.0)
    MAX_YAW_LEAD_END = np.deg2rad(48.0)

    VERTICAL_AMPLITUDE = 0.36
    VERTICAL_OMEGA = 0.75
    RECENTER_GAIN = 0.12
    LAST_SEEN_HEIGHT_GAIN = 0.55
    HEIGHT_BLEND = 0.35
    DIRECTION_BIAS_THRESHOLD = 0.04
