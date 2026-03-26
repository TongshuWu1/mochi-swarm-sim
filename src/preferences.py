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
    ENABLED_DEFAULT = True
    SHOW_HUD_DEFAULT = True
    SHOW_FIELD_WINDOW_DEFAULT = True
    FIELD_WINDOW_NAME = "Turbulence Field"
    FIELD_WINDOW_SIZE = 720

    BLIMP_BODY_NAME = "assembly"

    # Base field strength (before scale drift)
    XY_FORCE_MIN = 0.010
    XY_FORCE_MAX = 0.10
    Z_FORCE_MIN = -0.01
    Z_FORCE_MAX = 0.01
    YAW_TORQUE_MIN = -0.0025
    YAW_TORQUE_MAX = 0.0025

    # Global slow random drift ranges
    SCALE_MIN = 0.85
    SCALE_MAX = 1.45
    ANGLE_MIN_DEG = -40.0
    ANGLE_MAX_DEG = 40.0
    TARGET_RESAMPLE_MIN_S = 2.0
    TARGET_RESAMPLE_MAX_S = 5.0
    DRIFT_TIME_CONSTANT_S = 1.8

    # Spatial map bounds for visualization and field normalization
    FIELD_X_MIN = -14.0
    FIELD_X_MAX = 14.0
    FIELD_Y_MIN = -18.0
    FIELD_Y_MAX = 18.0
    FIELD_GRID_NX = 17
    FIELD_GRID_NY = 21


__all__ = ["CONTROL", "CONTROL_LIMITS", "CAMERA", "TURBULENCE"]
