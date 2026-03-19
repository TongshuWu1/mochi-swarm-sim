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


__all__ = ["CONTROL", "CONTROL_LIMITS", "CAMERA"]
