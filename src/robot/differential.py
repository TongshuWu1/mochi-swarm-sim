from enum import IntEnum, auto

import numpy as np

from src.definitions import State
from src.preferences import CONTROL, CONTROL_LIMITS
from src.state.robot_state import Behavior


class Control(IntEnum):
    FX = auto()
    FZ = auto()
    TX = auto()
    TZ = auto()


class Differential:
    def __init__(self):
        self.z_integral = 0.0
        self.yaw_integral = 0.0
        self.yawrate_integral = 0.0
        self._last_control_time: float | None = None

    def _dt(self, sim_time: float | None) -> float:
        if sim_time is None:
            return CONTROL.DT_FALLBACK
        if self._last_control_time is None:
            self._last_control_time = float(sim_time)
            return CONTROL.DT_FALLBACK
        dt = max(1e-4, float(sim_time) - self._last_control_time)
        self._last_control_time = float(sim_time)
        return dt

    def control(self, sensors: np.ndarray, behavior_commands: np.ndarray, sim_time: float | None = None) -> np.ndarray:
        if behavior_commands[Behavior.READY] == 0:
            return np.array([0.0, 0.0, np.pi / 2])

        target_forces = self._add_feedback(sensors, behavior_commands, dt=self._dt(sim_time))
        actuator_outputs = self._get_outputs(target_forces)
        return actuator_outputs

    def _add_feedback(self, sensors: np.ndarray, behavior_commands: np.ndarray, dt: float) -> dict:
        fx = behavior_commands[Behavior.FX_FORWARD]
        z_setpoint = behavior_commands[Behavior.Z_HEIGHT]
        tx = behavior_commands[Behavior.TX_ROLL]
        yaw_setpoint = behavior_commands[Behavior.Z_YAW]

        fz_out = 0.0
        tz_out = 0.0

        if CONTROL.Z_EN:
            e_z = z_setpoint - sensors[State.Z_ALTITUDE]
            self.z_integral += e_z * dt * CONTROL.KIZ
            self.z_integral = np.clip(self.z_integral, CONTROL.Z_INT_LOW, CONTROL.Z_INT_HIGH)
            fz_out = (e_z * CONTROL.KPZ) - (sensors[State.Z_ALTITUDE_VEL] * CONTROL.KDZ) + self.z_integral

        if CONTROL.YAW_EN:
            e_yaw = yaw_setpoint - sensors[State.Z_YAW]
            e_yaw = np.atan2(np.sin(e_yaw), np.cos(e_yaw))
            e_yaw = np.clip(e_yaw, -CONTROL_LIMITS.YAW_ERROR_MAX, CONTROL_LIMITS.YAW_ERROR_MAX)

            self.yaw_integral += e_yaw * dt * CONTROL.KIYAW
            self.yaw_integral = np.clip(self.yaw_integral, CONTROL_LIMITS.YAW_INTEGRAL_MIN, CONTROL_LIMITS.YAW_INTEGRAL_MAX)

            yaw_desired_rate = e_yaw * CONTROL.KPYAW + self.yaw_integral
            e_yawrate = yaw_desired_rate - sensors[State.Z_YAW_RATE]
            tz_out = yaw_desired_rate * CONTROL.KPPYAW + e_yawrate * CONTROL.KDYAW - sensors[State.Z_YAW_RATE] * CONTROL.KDDYAW

        return {Control.FX: fx, Control.FZ: fz_out, Control.TX: tx, Control.TZ: tz_out}

    def _get_outputs(self, feedback_controls: dict) -> np.ndarray:
        fx_target = float(np.clip(feedback_controls[Control.FX], CONTROL_LIMITS.FX_MIN, CONTROL_LIMITS.FX_MAX))
        fz_target = float(np.clip(feedback_controls[Control.FZ], CONTROL_LIMITS.FZ_MIN, CONTROL_LIMITS.FZ_MAX))
        tz_target = float(np.clip(feedback_controls[Control.TZ], CONTROL_LIMITS.TZ_MIN, CONTROL_LIMITS.TZ_MAX))
        l = CONTROL.LX

        theta = float(np.arctan2(fz_target, max(abs(fx_target), 0.06)))
        theta = float(np.clip(theta, CONTROL_LIMITS.SERVO_ANGLE_MIN, CONTROL_LIMITS.SERVO_ANGLE_MAX))

        forward_component = max(abs(fx_target), 0.0)
        vertical_component = abs(fz_target)
        thrust_total = np.hypot(forward_component, vertical_component)
        thrust_total = float(np.clip(thrust_total, 0.0, CONTROL_LIMITS.THRUST_TOTAL_MAX))

        c = np.cos(theta)
        yaw_term = 0.0 if abs(c) < 1e-5 else tz_target / (l * c)
        yaw_term = float(np.clip(yaw_term, CONTROL_LIMITS.YAW_TERM_MIN, CONTROL_LIMITS.YAW_TERM_MAX))

        f1 = 0.5 * (thrust_total - yaw_term)
        f2 = 0.5 * (thrust_total + yaw_term)

        if fx_target < 0.0:
            theta = np.pi - theta if theta >= 0.0 else -np.pi - theta

        f1_out = float(np.clip(f1, 0.0, 1.0))
        f2_out = float(np.clip(f2, 0.0, 1.0))
        theta_out = float(np.clip(theta, -np.pi, np.pi))
        return np.array([f1_out, f2_out, theta_out])
