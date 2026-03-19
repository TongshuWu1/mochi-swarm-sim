from typing import Tuple

import numpy as np

from src.definitions import Action, State

from .manual_preferences import MANUAL
from .robot_state import Behavior, RobotState


class ManualState(RobotState):
    def __init__(self):
        super().__init__()
        self.target_height = MANUAL.DEFAULT_HEIGHT
        self.target_yaw = 0.0
        self.target_thrust = 0.0
        self._initialized = False
        self._last_update_time: float | None = None

    @staticmethod
    def _wrap(angle: float) -> float:
        return float(np.atan2(np.sin(angle), np.cos(angle)))

    def _dt(self, sim_time: float) -> float:
        if self._last_update_time is None:
            self._last_update_time = float(sim_time)
            return 0.0
        dt = max(0.0, float(sim_time) - self._last_update_time)
        self._last_update_time = float(sim_time)
        return dt

    def update(
        self,
        sensors: np.ndarray,
        action_states: dict,
        tracking_result=None,
        sim_time: float = 0.0,
    ) -> Tuple[np.ndarray, RobotState]:
        behavior_targets = np.zeros(Behavior.NUM_PARAMS)
        behavior_targets[Behavior.READY] = 1.0 if action_states[Action.ARMED] else 0.0

        current_altitude = float(sensors[State.Z_ALTITUDE])
        current_yaw = float(sensors[State.Z_YAW])
        dt = self._dt(sim_time)

        if not self._initialized:
            self.target_height = current_altitude
            self.target_yaw = current_yaw
            self._initialized = True

        if action_states[Action.LEFT]:
            self.target_yaw += MANUAL.YAW_RATE_CMD * dt
        elif action_states[Action.RIGHT]:
            self.target_yaw -= MANUAL.YAW_RATE_CMD * dt
        self.target_yaw = self._wrap(self.target_yaw)

        if action_states[Action.UP]:
            self.target_height += MANUAL.CLIMB_RATE_CMD * dt
        elif action_states[Action.DOWN]:
            self.target_height -= MANUAL.CLIMB_RATE_CMD * dt
        self.target_height = float(np.clip(
            self.target_height,
            MANUAL.MIN_HEIGHT,
            MANUAL.MAX_HEIGHT,
        ))

        if action_states[Action.FORWARD]:
            self.target_thrust = MANUAL.FORWARD_CMD
        elif action_states[Action.BACKWARD]:
            self.target_thrust = MANUAL.BACKWARD_CMD
        else:
            self.target_thrust = 0.0

        behavior_targets[Behavior.Z_HEIGHT] = self.target_height
        behavior_targets[Behavior.Z_YAW] = self.target_yaw
        behavior_targets[Behavior.FX_FORWARD] = self.target_thrust
        return behavior_targets, self
