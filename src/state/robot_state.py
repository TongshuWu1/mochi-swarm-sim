from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Tuple, Optional
import numpy as np


class Behavior(IntEnum):
    READY = 0
    FX_FORWARD = auto()
    Z_HEIGHT = auto()
    TX_ROLL = auto()
    Z_YAW = auto()
    NUM_PARAMS = auto()


class RobotState(ABC):
    def __init__(self) -> None:
        # Standardized targets every state should expose for UI/telemetry
        # and downstream controllers. Subclasses should update these each cycle.
        self.target_height: float = 0.0
        self.target_yaw: float = 0.0
        self.target_thrust: float = 0.0
        self.expected_color: Optional[str] = None
        self.expected_shape: Optional[str] = None
        self.expected_label: Optional[str] = None

    @abstractmethod
    def update(
        self,
        sensors: np.ndarray,
        action_states: dict,
        tracking_result=None,
        sim_time: float = 0.0,
    ) -> Tuple[np.ndarray, "RobotState"]:
        """
        Updates the state logic.

        Args:
            sensors: A numpy array of current sensor readings.
            action_states: A dictionary of the current action states.

        Returns:
            A tuple containing:
            1. A numpy array of high-level behavior commands
               (e.g., [ready, Fx, Fz, Tx, Tz]).
            2. The next state (can be 'self' to remain in the current state).
        """
        pass
