from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .auto_preferences import AUTO, AUTO_SEARCH


@dataclass
class SpiralSearchCommand:
    target_yaw: float
    target_height: float
    target_thrust: float
    debug_status: str


class SpiralSearchController:
    """Continuous transit + true spiral reacquisition helper."""

    def __init__(self):
        self.direction = 1.0
        self.phase = "transit"
        self.started_time = 0.0
        self.phase_started_time = 0.0
        self.cycle_index = 0

        self._yaw_prev_wrapped: Optional[float] = None
        self._yaw_unwrapped = 0.0
        self._yaw_start_unwrapped = 0.0

        self.target_height = AUTO.DEFAULT_HEIGHT
        self.transit_steps_remaining = 0

    @staticmethod
    def wrap(angle: float) -> float:
        return float(np.atan2(np.sin(angle), np.cos(angle)))

    def unwrap_yaw(self, yaw_wrapped: float) -> float:
        if self._yaw_prev_wrapped is None:
            self._yaw_prev_wrapped = yaw_wrapped
            self._yaw_unwrapped = yaw_wrapped
            return self._yaw_unwrapped

        delta = np.atan2(
            np.sin(yaw_wrapped - self._yaw_prev_wrapped),
            np.cos(yaw_wrapped - self._yaw_prev_wrapped),
        )
        self._yaw_unwrapped += delta
        self._yaw_prev_wrapped = yaw_wrapped
        return self._yaw_unwrapped

    def _choose_direction(self, last_seen_dx: float) -> None:
        if abs(last_seen_dx) > AUTO_SEARCH.DIRECTION_BIAS_THRESHOLD:
            self.direction = 1.0 if last_seen_dx > 0.0 else -1.0
        elif AUTO_SEARCH.ALTERNATE_DIRECTION_EACH_CYCLE and (self.cycle_index % 2 == 1):
            self.direction *= -1.0

    def _choose_target_height(
        self,
        current_height: float,
        nominal_height: float,
        last_seen_dy: float,
    ) -> float:
        desired_height = nominal_height - AUTO_SEARCH.LAST_SEEN_HEIGHT_GAIN * last_seen_dy
        return float(np.clip(
            (1.0 - AUTO_SEARCH.HEIGHT_BLEND) * current_height
            + AUTO_SEARCH.HEIGHT_BLEND * desired_height,
            AUTO.MIN_HEIGHT,
            AUTO.MAX_HEIGHT,
        ))

    def _start_transit(
        self,
        current_yaw_unwrapped: float,
        sim_time: float,
        current_height: float,
        nominal_height: float,
        last_seen_dx: float,
        last_seen_dy: float,
        dt_estimate: float,
    ) -> None:
        self.phase = "transit"
        self.phase_started_time = sim_time
        self._yaw_start_unwrapped = current_yaw_unwrapped
        self._choose_direction(last_seen_dx)
        self.target_height = self._choose_target_height(current_height, nominal_height, last_seen_dy)
        safe_dt = max(float(dt_estimate), 1e-4)
        self.transit_steps_remaining = int(max(1, round(AUTO_SEARCH.TRANSIT_DURATION / safe_dt)))
        self.cycle_index += 1

    def begin(
        self,
        current_yaw_wrapped: float,
        current_yaw_unwrapped: float,
        sim_time: float,
        current_height: float,
        nominal_height: float,
        last_seen_dx: float,
        last_seen_dy: float,
        dt_estimate: float,
    ) -> SpiralSearchCommand:
        self.started_time = sim_time
        self._start_transit(
            current_yaw_unwrapped=current_yaw_unwrapped,
            sim_time=sim_time,
            current_height=current_height,
            nominal_height=nominal_height,
            last_seen_dx=last_seen_dx,
            last_seen_dy=last_seen_dy,
            dt_estimate=dt_estimate,
        )
        return SpiralSearchCommand(
            target_yaw=current_yaw_wrapped,
            target_height=self.target_height,
            target_thrust=AUTO_SEARCH.FORWARD_THRUST_TRANSIT,
            debug_status=f"search:init dir={'CCW' if self.direction > 0 else 'CW'}",
        )

    def step(
        self,
        sim_time: float,
        current_yaw_wrapped: float,
        current_yaw_unwrapped: float,
        current_height: float,
        nominal_height: float,
        label: str | None,
        dt_estimate: float,
    ) -> SpiralSearchCommand:
        if self.phase == "transit":
            self.transit_steps_remaining -= 1
            if self.transit_steps_remaining <= 0:
                self.phase = "spiral"
                self.phase_started_time = sim_time
                self._yaw_start_unwrapped = current_yaw_unwrapped
            return SpiralSearchCommand(
                target_yaw=current_yaw_wrapped,
                target_height=self.target_height,
                target_thrust=AUTO_SEARCH.FORWARD_THRUST_TRANSIT,
                debug_status=(
                    f"search:transit {label} | z={self.target_height:.2f} | "
                    f"dir={'CCW' if self.direction > 0 else 'CW'}"
                ),
            )

        traveled_along_dir = self.direction * (current_yaw_unwrapped - self._yaw_start_unwrapped)
        remaining_along_dir = AUTO_SEARCH.SPIRAL_TURN_TOTAL - traveled_along_dir

        if remaining_along_dir <= AUTO_SEARCH.SPIRAL_COMPLETE_MARGIN:
            self._start_transit(
                current_yaw_unwrapped=current_yaw_unwrapped,
                sim_time=sim_time,
                current_height=current_height,
                nominal_height=nominal_height,
                last_seen_dx=0.0,
                last_seen_dy=0.0,
                dt_estimate=dt_estimate,
            )
            return SpiralSearchCommand(
                target_yaw=current_yaw_wrapped,
                target_height=self.target_height,
                target_thrust=AUTO_SEARCH.FORWARD_THRUST_TRANSIT,
                debug_status=(
                    f"search:reset {label} | z={self.target_height:.2f} | "
                    f"dir={'CCW' if self.direction > 0 else 'CW'}"
                ),
            )

        frac_remaining = float(np.clip(
            remaining_along_dir / max(AUTO_SEARCH.SPIRAL_TURN_TOTAL, 1e-6),
            0.0,
            1.0,
        ))
        step_size = max(AUTO_SEARCH.SPIRAL_TAPER_MIN, frac_remaining) * AUTO_SEARCH.SPIRAL_YAW_STEP
        step = min(step_size, max(0.0, remaining_along_dir))
        target_yaw_unwrapped = current_yaw_unwrapped + self.direction * step
        target_yaw = self.wrap(target_yaw_unwrapped)

        search_age = sim_time - self.started_time
        z_offset = AUTO_SEARCH.VERTICAL_AMPLITUDE * np.sin(AUTO_SEARCH.VERTICAL_OMEGA * search_age)
        target_height = float(np.clip(
            self.target_height + z_offset + AUTO_SEARCH.RECENTER_GAIN * (nominal_height - current_height),
            AUTO.MIN_HEIGHT,
            AUTO.MAX_HEIGHT,
        ))

        return SpiralSearchCommand(
            target_yaw=target_yaw,
            target_height=target_height,
            target_thrust=AUTO_SEARCH.FORWARD_THRUST_SPIRAL,
            debug_status=(
                f"search:spiral {label} | rem={np.rad2deg(remaining_along_dir):.0f}deg | "
                f"step={np.rad2deg(step):.1f}deg | z={target_height:.2f} | "
                f"dir={'CCW' if self.direction > 0 else 'CW'}"
            ),
        )
