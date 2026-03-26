"""Global project preferences.

State-specific tuning lives next to the corresponding state implementation to avoid
surprising cross-module caching behavior. Keep only truly global knobs here.
"""

from __future__ import annotations

import numpy as np


class CONTROL:
    Z_EN = True
    YAW_EN = True

    KPZ = 0.65
    KDZ = 0.45
    KIZ = 0.0
    Z_INT_LOW = 0.0
    Z_INT_HIGH = 0.12

    KPYAW = 1.0
    KPPYAW = 0.02
    KDYAW = 0.02
    KDDYAW = 0.015
    KIYAW = 0.0

    LX = 0.1
    DT_FALLBACK = 0.01


class CONTROL_LIMITS:
    YAW_ERROR_MAX = np.pi / 4
    YAW_INTEGRAL_MIN = -0.30
    YAW_INTEGRAL_MAX = 0.30

    FX_MIN = -0.35
    FX_MAX = 0.35
    FZ_MIN = -1.4
    FZ_MAX = 1.4
    TZ_MIN = -0.08
    TZ_MAX = 0.08

    SERVO_ANGLE_MIN = -1.45
    SERVO_ANGLE_MAX = 1.45
    THRUST_TOTAL_MAX = 0.72
    YAW_TERM_MIN = -0.26
    YAW_TERM_MAX = 0.26


class CAMERA:
    RESOLUTION_NAME = "HQVGA"
    FPS = 10.0
    PROCESSING_MODE = "RGB"  # RGB, GRAY, HSV
    SHOW_PROCESSED_WINDOW = True
    WINDOW_SCALE = 2
    FIXED_CAMERA_NAME = "nicla_vision"
    FOLLOW_BODY_NAME = "assembly"



class TURBULENCE:
    ENABLED_BY_DEFAULT = False
    SEED = 7

    FIELD_X_MIN = -40.0
    FIELD_X_MAX = 65.0
    FIELD_Y_MIN = -12.0
    FIELD_Y_MAX = 12.0

    HORIZONTAL_MAG_MIN = 0.012
    HORIZONTAL_MAG_MAX = 0.045
    VERTICAL_MAG_MAX = 0.005
    YAW_TORQUE_GAIN = 0.055
    YAW_TORQUE_MAX = 0.0035

    BASE_HEADING_DEG = 6.0
    X_HEADING_VARIATION_DEG = 24.0
    Y_HEADING_VARIATION_DEG = 18.0

    SCALE_MIN = 0.85
    SCALE_MAX = 1.20
    MAX_ANGLE_OFFSET_DEG = 14.0
    TARGET_HOLD_TIME_MIN = 2.0
    TARGET_HOLD_TIME_MAX = 5.0
    SMOOTHING_TIME_CONSTANT = 2.8

    FIELD_GRID_COLS = 13
    FIELD_GRID_ROWS = 7
    FIELD_ARROW_SCALE = 560.0
    WINDOW_NAME = "Turbulence Field"
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 420


__all__ = ["CONTROL", "CONTROL_LIMITS", "CAMERA", "TURBULENCE"]
