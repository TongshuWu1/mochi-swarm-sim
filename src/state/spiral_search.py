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
    """Direct spiral reacquisition helper.

    The key behavior here is that search uses an internal commanded yaw that moves
    smoothly in time. We do *not* regenerate the yaw target directly from the
    instantaneous measured yaw each tick, because that makes the target lurch with
    sensor/controller noise and causes visible jitter.
    """

    def __init__(self):
        self.direction = 1.0
        self.started_time = 0.0

        self._yaw_prev_wrapped: Optional[float] = None
        self._yaw_unwrapped = 0.0
        self._yaw_start_unwrapped = 0.0
        self._command_yaw_unwrapped = 0.0

        self.target_height = AUTO.DEFAULT_HEIGHT

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return float(a + (b - a) * np.clip(t, 0.0, 1.0))

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
        else:
            # Keep the previous direction instead of alternating mid-search.
            self.direction = 1.0 if self.direction >= 0.0 else -1.0

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

    def _start_session(
        self,
        current_yaw_unwrapped: float,
        sim_time: float,
        current_height: float,
        nominal_height: float,
        last_seen_dx: float,
        last_seen_dy: float,
    ) -> None:
        self.started_time = sim_time
        self._yaw_start_unwrapped = current_yaw_unwrapped
        self._command_yaw_unwrapped = current_yaw_unwrapped
        self._choose_direction(last_seen_dx)
        self.target_height = self._choose_target_height(current_height, nominal_height, last_seen_dy)

    @staticmethod
    def _safe_dt(dt_estimate: float) -> float:
        return float(np.clip(dt_estimate, 1e-3, 0.05))

    @staticmethod
    def _search_progress(search_age: float) -> float:
        linear = np.clip(search_age / max(AUTO_SEARCH.SPIRAL_RATE_DECAY_TIME, 1e-6), 0.0, 1.0)
        # Ease-out: keep the turn rate high initially, then taper it down smoothly.
        return float(1.0 - (1.0 - linear) ** 2)

    def _advance_command_yaw(
        self,
        sim_time: float,
        current_yaw_unwrapped: float,
        dt_estimate: float,
    ) -> tuple[float, float, float, float, float]:
        search_age = max(0.0, sim_time - self.started_time)
        expand_progress = self._search_progress(search_age)

        yaw_rate = self._lerp(
            AUTO_SEARCH.SPIRAL_INITIAL_YAW_RATE,
            AUTO_SEARCH.SPIRAL_FINAL_YAW_RATE,
            expand_progress,
        )
        dt = self._safe_dt(dt_estimate)
        step = yaw_rate * dt

        proposed_command = self._command_yaw_unwrapped + self.direction * step

        max_lead = self._lerp(
            AUTO_SEARCH.MAX_YAW_LEAD_START,
            AUTO_SEARCH.MAX_YAW_LEAD_END,
            expand_progress,
        )
        lead = self.direction * (proposed_command - current_yaw_unwrapped)
        if lead > max_lead:
            proposed_command = current_yaw_unwrapped + self.direction * max_lead
            step = max(0.0, self.direction * (proposed_command - self._command_yaw_unwrapped))
        elif lead < 0.0:
            # Never command a heading that falls behind the chosen search direction.
            proposed_command = current_yaw_unwrapped
            step = max(0.0, self.direction * (proposed_command - self._command_yaw_unwrapped))

        self._command_yaw_unwrapped = proposed_command
        return self._command_yaw_unwrapped, step, expand_progress, yaw_rate, search_age

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
        del current_yaw_wrapped
        self._start_session(
            current_yaw_unwrapped=current_yaw_unwrapped,
            sim_time=sim_time,
            current_height=current_height,
            nominal_height=nominal_height,
            last_seen_dx=last_seen_dx,
            last_seen_dy=last_seen_dy,
        )
        command_yaw_unwrapped, step, growth, yaw_rate, _ = self._advance_command_yaw(
            sim_time=sim_time,
            current_yaw_unwrapped=current_yaw_unwrapped,
            dt_estimate=dt_estimate,
        )
        initial_target_yaw = self.wrap(command_yaw_unwrapped)
        initial_thrust = AUTO_SEARCH.SPIRAL_THRUST
        return SpiralSearchCommand(
            target_yaw=initial_target_yaw,
            target_height=self.target_height,
            target_thrust=initial_thrust,
            debug_status=(
                f"search:spiral {self._label_text(None)} | "
                f"rate={np.rad2deg(yaw_rate):.1f}dps | step={np.rad2deg(step):.1f}deg | "
                f"thrust={initial_thrust:.3f} | grow={growth:.2f} | z={self.target_height:.2f} | "
                f"dir={'CCW' if self.direction > 0 else 'CW'}"
            ),
        )

    @staticmethod
    def _label_text(label: str | None) -> str:
        return label if label else "target"

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
        del current_yaw_wrapped

        command_yaw_unwrapped, step, expand_progress, yaw_rate, search_age = self._advance_command_yaw(
            sim_time=sim_time,
            current_yaw_unwrapped=current_yaw_unwrapped,
            dt_estimate=dt_estimate,
        )
        target_yaw = self.wrap(command_yaw_unwrapped)

        z_offset = AUTO_SEARCH.VERTICAL_AMPLITUDE * np.sin(AUTO_SEARCH.VERTICAL_OMEGA * search_age)
        target_height = float(np.clip(
            self.target_height + z_offset + AUTO_SEARCH.RECENTER_GAIN * (nominal_height - current_height),
            AUTO.MIN_HEIGHT,
            AUTO.MAX_HEIGHT,
        ))
        target_thrust = AUTO_SEARCH.SPIRAL_THRUST

        lead_deg = np.rad2deg(self.direction * (command_yaw_unwrapped - current_yaw_unwrapped))
        return SpiralSearchCommand(
            target_yaw=target_yaw,
            target_height=target_height,
            target_thrust=target_thrust,
            debug_status=(
                f"search:spiral {self._label_text(label)} | lead={lead_deg:.1f}deg | "
                f"rate={np.rad2deg(yaw_rate):.1f}dps | step={np.rad2deg(step):.1f}deg | "
                f"thrust={target_thrust:.3f} | grow={expand_progress:.2f} | z={target_height:.2f} | "
                f"dir={'CCW' if self.direction > 0 else 'CW'}"
            ),
        )
