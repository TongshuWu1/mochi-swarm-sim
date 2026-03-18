from typing import Tuple
import numpy as np
from .robot_state import RobotState, Behavior
from src.definitions import Action, State
from src import preferences


class ExplorationState(RobotState):
    """
    Autonomous exploration: perform repeated spirals at different heights.
    """

    def __init__(self):
        super().__init__()
        # Spiral parameters/state
        self.target_height: float = preferences.B_DEFAULT_HEIGHT
        self.final_target_yaw: float = 2 * np.pi  # total radians to rotate this spiral
        self._yaw_direction: int = 1  # +1 CCW, -1 CW (convention arbitrary)

        # Unwrapped yaw tracking
        self._yaw_unwrapped: float = 0.0
        self._yaw_prev_wrapped: float | None = None
        self._yaw_start_unwrapped: float = 0.0
        # We no longer store an absolute goal on the unwrapped axis; instead we
        # keep the start and total sweep (final_target_yaw) and compute remaining
        # on the fly.

        # Incremental setpoint step (rad) per control cycle
        self.yaw_step: float = np.pi / 8  # base step target (scaled by taper)
        self.taper_min: float = 0.3  # keep at least 30% of base step near the end
        # Forward thrust during spiral motion (normalized [-1,1])
        self.target_thrust: float = 0.4

        # Transit phase parameters (drive straight before turning)
        self.transit_duration_s: float = 5.0
        self.transit_thrust: float = self.target_thrust
        self._in_transit: bool = False
        self._transit_steps_remaining: int = 0
        self._eps: float = 1e-3
        self.target_yaw: float = 0.0

        # Initialization flag
        self._initialized: bool = False

    def _unwrap_yaw(self, yaw_wrapped: float) -> float:
        """Maintain a continuous yaw estimate from a wrapped measurement [-π, π]."""
        if self._yaw_prev_wrapped is None:
            self._yaw_prev_wrapped = yaw_wrapped
            self._yaw_unwrapped = yaw_wrapped
            return self._yaw_unwrapped

        # Smallest signed angle difference (-π, π]
        delta = np.atan2(
            np.sin(yaw_wrapped - self._yaw_prev_wrapped),
            np.cos(yaw_wrapped - self._yaw_prev_wrapped),
        )
        self._yaw_unwrapped += delta
        self._yaw_prev_wrapped = yaw_wrapped
        return self._yaw_unwrapped

    def _start_new_spiral(self, current_yaw_unwrapped: float):
        # Pick new altitude and total rotation
        low = preferences.B_DEFAULT_HEIGHT - preferences.B_DEFAULT_RANGE
        high = preferences.B_DEFAULT_HEIGHT + preferences.B_DEFAULT_RANGE
        self.target_height = np.random.uniform(low, high)

        self.final_target_yaw = np.random.uniform(2 * np.pi, 4 * np.pi)
        self._yaw_direction = np.random.choice([-1, 1])

        self._yaw_start_unwrapped = current_yaw_unwrapped
        # Begin with a short straight-line transit before turning
        self._in_transit = True
        # Use a step counter to avoid floating-time accumulation issues
        steps = int(max(1, round(self.transit_duration_s / preferences.DT)))
        self._transit_steps_remaining = steps

    def update(
        self,
        sensors: np.ndarray,
        action_states: dict,
        tracking_result=None,
        sim_time: float = 0.0,
    ) -> Tuple[np.ndarray, RobotState]:

        behavior_targets = np.zeros(Behavior.NUM_PARAMS)

        # READY follows ARMED input (or set to 1.0 to always run)
        behavior_targets[Behavior.READY] = 1.0 if action_states[Action.ARMED] else 0.0

        # Unwrap the current yaw measurement
        yaw_wrapped = sensors[State.Z_YAW]
        yaw_unwrapped = self._unwrap_yaw(yaw_wrapped)

        # Initialize on first call
        if not self._initialized:
            self._start_new_spiral(yaw_unwrapped)
            self._initialized = True

        # Compute remaining rotation along the commanded direction
        traveled_along_dir = self._yaw_direction * (
            yaw_unwrapped - self._yaw_start_unwrapped
        )
        remaining_along_dir = self.final_target_yaw - traveled_along_dir
        spiral_complete = remaining_along_dir <= self._eps

        if spiral_complete:
            # Start a new spiral at a new height and yaw sweep (with transit phase)
            self._start_new_spiral(yaw_unwrapped)
            # Recompute remaining after reset
            traveled_along_dir = 0.0
            remaining_along_dir = self.final_target_yaw

        # Transit phase: hold yaw for a short duration and drive forward
        if self._in_transit:
            # Decrement transit step budget and exit when done
            self._transit_steps_remaining -= 1
            if self._transit_steps_remaining <= 0:
                self._in_transit = False
            # Hold current yaw (no turn) during transit
            target_yaw = yaw_wrapped
            fx_cmd = self.transit_thrust
        else:
            # Tapered yaw step: faster at start, slower near the goal
            frac_remaining = float(
                np.clip(remaining_along_dir / self.final_target_yaw, 0.0, 1.0)
            )
            step_size = max(self.taper_min, frac_remaining) * self.yaw_step
            step = min(step_size, max(0.0, remaining_along_dir))
            target_yaw_unwrapped = yaw_unwrapped + self._yaw_direction * step
            # Wrap to [-π, π] for the controller input
            target_yaw = np.atan2(
                np.sin(target_yaw_unwrapped), np.cos(target_yaw_unwrapped)
            )
            fx_cmd = self.target_thrust
        # Populate targets: altitude fixed for current spiral, yaw incremental target,
        # forward thrust to promote spiral motion.
        behavior_targets[Behavior.Z_HEIGHT] = self.target_height
        behavior_targets[Behavior.Z_YAW] = target_yaw
        behavior_targets[Behavior.FX_FORWARD] = fx_cmd

        self.target_yaw = target_yaw

        return behavior_targets, self
