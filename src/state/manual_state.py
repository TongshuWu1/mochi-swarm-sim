from typing import Tuple
import numpy as np
from .robot_state import RobotState, Behavior
from src.definitions import Action, State


class ManualState(RobotState):
    def __init__(self):
        super().__init__()
        self.target_height = 1.5
        self.target_yaw = 0.0  # angle
        self.target_thrust = 0.0
        self._prev_updown = False  #  overriding feedback during manual input
        self._prev_leftright = False  #  overriding feedback during manual input

    def update(
        self,
        sensors: np.ndarray,
        action_states: dict,
        tracking_result=None,
        sim_time: float = 0.0,
    ) -> Tuple[np.ndarray, RobotState]:

        # equivalent to 'behave.params' in DiffController.ino
        behavior_targets = np.zeros(Behavior.NUM_PARAMS)

        # translate key presses into desired forces

        behavior_targets[Behavior.READY] = 1.0 if action_states[Action.ARMED] else 0.0

        # Set default targets for the PID controller
        behavior_targets[Behavior.Z_HEIGHT] = self.target_height
        behavior_targets[Behavior.Z_YAW] = self.target_yaw

        # up/down altitude
        if action_states[Action.UP]:
            self.target_height = sensors[State.Z_ALTITUDE] + 3.0
            self._prev_updown = True
        elif action_states[Action.DOWN]:
            self.target_height = sensors[State.Z_ALTITUDE] - 3.0
            self._prev_updown = True
        elif self._prev_updown:
            self.target_height = sensors[State.Z_ALTITUDE]
            self._prev_updown = False
        behavior_targets[Behavior.Z_HEIGHT] = self.target_height

        # forward/backward thrust
        if action_states[Action.FORWARD]:
            self.target_thrust = 1.0
        elif action_states[Action.BACKWARD]:
            self.target_thrust = -1.0
        else:
            self.target_thrust = 0.0
        behavior_targets[Behavior.FX_FORWARD] = self.target_thrust

        # left/right yaw
        if action_states[Action.LEFT]:
            self.target_yaw = sensors[State.Z_YAW] + (np.pi / 2)
            self._prev_leftright = True
        elif action_states[Action.RIGHT]:
            self.target_yaw = sensors[State.Z_YAW] - (np.pi / 2)
            self._prev_leftright = True
        elif self._prev_leftright:
            self.target_yaw = sensors[State.Z_YAW]
            self._prev_leftright = False
        # Normalize yaw to [-pi, pi]
        self.target_yaw = np.atan2(np.sin(self.target_yaw), np.cos(self.target_yaw))
        behavior_targets[Behavior.Z_YAW] = self.target_yaw

        # Always stay in ManualState in this example
        return behavior_targets, self
