from __future__ import annotations

from typing import Tuple

import numpy as np

from src.definitions import Action, State

from .auto_preferences import AUTO, AUTO_CHARGE, AUTO_SEARCH, AUTO_TRACK
from .robot_state import Behavior, RobotState
from .spiral_search import SpiralSearchController


class AutoGateSequenceState(RobotState):
    """Vision-guided autonomous sequence for touching colored balloons."""

    def __init__(self):
        super().__init__()
        self.sequence_index = 0

        self.target_height = AUTO.DEFAULT_HEIGHT
        self.target_yaw = 0.0
        self.target_thrust = 0.0

        self.mode = "init"
        self.completed = False
        self.debug_status = "init"

        self.filtered_dx = 0.0
        self.filtered_dy = 0.0
        self.filtered_width_frac = 0.0
        self.filtered_height_frac = 0.0
        self.filtered_area = 0.0
        self.last_vision_timestamp = -1e9
        self.last_pass_time = -1e9
        self.last_detection_time = -1e9
        self.charge_started_time = -1e9
        self.charge_seen_counter = 0
        self.charge_candidate_started_time = -1e9
        self.charge_hold_yaw = 0.0
        self.charge_hold_height = 0.0
        self.last_seen_dx = 0.0
        self.last_seen_dy = 0.0
        self.search_nominal_height = AUTO.DEFAULT_HEIGHT
        self.search = SpiralSearchController()
        self._last_update_time: float | None = None

        self._sync_expected()
        self._enter_search_mode(0.0, 0.0, 0.0, AUTO.DEFAULT_HEIGHT)

    @property
    def gate_sequence(self) -> list[str]:
        return list(AUTO.TARGET_SEQUENCE)

    @property
    def target_height_hints(self) -> list[float]:
        return list(AUTO.TARGET_HEIGHT_HINTS)

    @staticmethod
    def _wrap(angle: float) -> float:
        return float(np.atan2(np.sin(angle), np.cos(angle)))

    def _dt(self, sim_time: float) -> float:
        if self._last_update_time is None:
            self._last_update_time = float(sim_time)
            return 0.01
        dt = max(1e-4, float(sim_time) - self._last_update_time)
        self._last_update_time = float(sim_time)
        return dt

    def _sync_expected(self):
        seq = self.gate_sequence
        if self.sequence_index < len(seq):
            self.expected_color = seq[self.sequence_index]
            self.expected_shape = None
            self.expected_label = f"{self.expected_color} balloon"
            nominals = self.target_height_hints
            self.search_nominal_height = float(np.clip(
                nominals[self.sequence_index],
                AUTO.MIN_HEIGHT,
                AUTO.MAX_HEIGHT,
            ))
        else:
            self.expected_color = None
            self.expected_shape = None
            self.expected_label = None
            self.completed = True

    def _reset_tracking_filters(self):
        self.filtered_dx = 0.0
        self.filtered_dy = 0.0
        self.filtered_width_frac = 0.0
        self.filtered_height_frac = 0.0
        self.filtered_area = 0.0
        self.charge_seen_counter = 0
        self.charge_candidate_started_time = -1e9

    def _is_new_vision(self, tracking_result) -> bool:
        return tracking_result is not None and float(tracking_result.timestamp) > self.last_vision_timestamp + 1e-9

    def _tracking_matches_expected(self, tracking_result, within_blind_window: bool) -> bool:
        return bool(
            tracking_result
            and tracking_result.found
            and tracking_result.color_name
            and tracking_result.color_name == self.expected_color
            and bool(getattr(tracking_result, "confidence", 0.0)) >= AUTO.MIN_TRACK_CONFIDENCE
        )

    def _enter_search_mode(self, current_yaw: float, current_yaw_unwrapped: float, sim_time: float, current_height: float):
        self.mode = "search"
        cmd = self.search.begin(
            current_yaw_wrapped=current_yaw,
            current_yaw_unwrapped=current_yaw_unwrapped,
            sim_time=sim_time,
            current_height=current_height,
            nominal_height=self.search_nominal_height,
            last_seen_dx=self.last_seen_dx,
            last_seen_dy=self.last_seen_dy,
            dt_estimate=self._dt(sim_time),
        )
        self.target_yaw = cmd.target_yaw
        self.target_height = cmd.target_height
        self.target_thrust = cmd.target_thrust
        self.charge_seen_counter = 0
        self.debug_status = f"{cmd.debug_status} {self.expected_label}"

    def _update_filtered_tracking(self, tracking_result):
        dx = float(tracking_result.offset_x_norm or 0.0)
        dy = float(tracking_result.offset_y_norm or 0.0)
        width_frac = float((tracking_result.width_px or 0.0) / max(float(tracking_result.frame_width or 1), 1.0))
        height_frac = float((tracking_result.height_px or 0.0) / max(float(tracking_result.frame_height or 1), 1.0))
        area = float(tracking_result.area_px or 0.0)

        self.filtered_dx = (1.0 - AUTO_TRACK.OFFSET_ALPHA) * self.filtered_dx + AUTO_TRACK.OFFSET_ALPHA * dx
        self.filtered_dy = (1.0 - AUTO_TRACK.OFFSET_ALPHA) * self.filtered_dy + AUTO_TRACK.OFFSET_ALPHA * dy
        self.filtered_width_frac = (1.0 - AUTO_TRACK.SIZE_ALPHA) * self.filtered_width_frac + AUTO_TRACK.SIZE_ALPHA * width_frac
        self.filtered_height_frac = (1.0 - AUTO_TRACK.SIZE_ALPHA) * self.filtered_height_frac + AUTO_TRACK.SIZE_ALPHA * height_frac
        self.filtered_area = (1.0 - AUTO_TRACK.SIZE_ALPHA) * self.filtered_area + AUTO_TRACK.SIZE_ALPHA * area

        self.last_seen_dx = self.filtered_dx
        self.last_seen_dy = self.filtered_dy
        return self.filtered_dx, self.filtered_dy, self.filtered_area, self.filtered_width_frac, self.filtered_height_frac

    @staticmethod
    def _target_close_enough(area: float, width_frac: float, height_frac: float) -> bool:
        return (
            area >= AUTO_CHARGE.AREA_THRESHOLD
            or width_frac >= AUTO_CHARGE.WIDTH_FRAC_THRESHOLD
            or height_frac >= AUTO_CHARGE.HEIGHT_FRAC_THRESHOLD
        )

    @staticmethod
    def _target_soft_close(area: float, width_frac: float, height_frac: float) -> bool:
        return (
            area >= AUTO_CHARGE.SOFT_AREA_THRESHOLD
            or width_frac >= AUTO_CHARGE.SOFT_WIDTH_FRAC_THRESHOLD
            or height_frac >= AUTO_CHARGE.SOFT_HEIGHT_FRAC_THRESHOLD
        )

    @staticmethod
    def _target_well_centered(center_error: float) -> bool:
        return center_error <= AUTO_CHARGE.ALIGNMENT_RADIUS

    def _begin_charge(self, hold_yaw: float, hold_height: float, sim_time: float):
        self.mode = "charge"
        self.charge_candidate_started_time = -1e9
        self.charge_started_time = sim_time
        self.charge_hold_yaw = self._wrap(hold_yaw)
        self.charge_hold_height = float(np.clip(hold_height, AUTO.MIN_HEIGHT, AUTO.MAX_HEIGHT))
        self.target_yaw = self.charge_hold_yaw
        self.target_height = self.charge_hold_height
        self.target_thrust = AUTO_CHARGE.RAMP_START_THRUST
        self.debug_status = f"charge:{self.expected_label}"

    def _advance_target(self, current_yaw: float, current_altitude: float, sim_time: float):
        self.sequence_index += 1
        self.last_pass_time = sim_time
        self.last_detection_time = -1e9
        self._reset_tracking_filters()
        self._sync_expected()
        if self.completed:
            self.mode = "complete"
            self.debug_status = "complete"
            self.target_thrust = 0.0
        else:
            self.debug_status = f"advance:{self.sequence_index + 1}/{len(self.gate_sequence)}->{self.expected_label}"
            self._enter_search_mode(current_yaw, self.search.unwrap_yaw(current_yaw), sim_time, current_altitude)

    def _run_track(self, current_yaw: float, current_altitude: float, sim_time: float, tracking_result):
        dx, dy, area, width_frac, height_frac = self._update_filtered_tracking(tracking_result)
        center_error = float(np.hypot(dx, dy))
        self.last_detection_time = sim_time
        self.mode = "track"

        yaw_step = float(np.clip(-AUTO_TRACK.YAW_GAIN * dx, -AUTO_TRACK.MAX_YAW_STEP, AUTO_TRACK.MAX_YAW_STEP))
        self.target_yaw = self._wrap(current_yaw + yaw_step)

        height_step = float(np.clip(-AUTO_TRACK.HEIGHT_GAIN * dy, -AUTO_TRACK.MAX_HEIGHT_STEP, AUTO_TRACK.MAX_HEIGHT_STEP))
        self.target_height = float(np.clip(current_altitude + height_step, AUTO.MIN_HEIGHT, AUTO.MAX_HEIGHT))

        if center_error < AUTO_CHARGE.TIGHT_CENTER_RADIUS:
            self.target_thrust = AUTO_TRACK.FORWARD_THRUST_NEAR
        elif center_error < 0.22:
            self.target_thrust = AUTO_TRACK.FORWARD_THRUST_MID
        else:
            self.target_thrust = AUTO_TRACK.FORWARD_THRUST_FAR

        close_enough = self._target_close_enough(area, width_frac, height_frac)
        soft_close = self._target_soft_close(area, width_frac, height_frac)
        well_centered = self._target_well_centered(center_error)
        nearly_centered = center_error <= (AUTO_CHARGE.ALIGNMENT_RADIUS * 1.45)

        charge_candidate = (well_centered and soft_close) or (nearly_centered and close_enough)
        if charge_candidate:
            if self.charge_candidate_started_time < -1e8:
                self.charge_candidate_started_time = sim_time
            self.charge_seen_counter += 1
        else:
            self.charge_seen_counter = 0
            self.charge_candidate_started_time = -1e9

        dwell_time = 0.0 if self.charge_candidate_started_time < -1e8 else (sim_time - self.charge_candidate_started_time)
        should_charge = (
            self.charge_seen_counter >= AUTO_CHARGE.CONFIRM_FRAMES
            or dwell_time >= AUTO_CHARGE.CONFIRM_TIME
        )

        if should_charge:
            self._begin_charge(hold_yaw=self.target_yaw, hold_height=self.target_height, sim_time=sim_time)
        else:
            self.debug_status = (
                f"track:{self.expected_label} {self.sequence_index + 1}/{len(self.gate_sequence)} | "
                f"err={center_error:.2f} area={area:.0f} w={width_frac:.2f} h={height_frac:.2f} "
                f"charge_n={self.charge_seen_counter} dwell={dwell_time:.2f}s"
            )

    def _run_charge(self, current_yaw: float, current_altitude: float, sim_time: float, expected_seen: bool):
        self.target_yaw = self.charge_hold_yaw
        self.target_height = self.charge_hold_height
        elapsed = sim_time - self.charge_started_time

        if elapsed < AUTO_CHARGE.RAMP_TIME:
            ramp = elapsed / max(AUTO_CHARGE.RAMP_TIME, 1e-6)
            self.target_thrust = (
                AUTO_CHARGE.RAMP_START_THRUST
                + (AUTO_CHARGE.THRUST - AUTO_CHARGE.RAMP_START_THRUST) * ramp
            )
        else:
            self.target_thrust = AUTO_CHARGE.THRUST

        # Finish the full charge no matter what.
        timed_out = elapsed >= AUTO_CHARGE.DURATION

        if timed_out:
            self._advance_target(current_yaw, current_altitude, sim_time)
            return

        remaining = AUTO_CHARGE.DURATION - elapsed
        seen_txt = "seen" if expected_seen else "blind"
        self.debug_status = f"charge:{self.expected_label} | {seen_txt} | t_left={remaining:.2f}s"

    def update(
        self,
        sensors: np.ndarray,
        action_states: dict,
        tracking_result=None,
        sim_time: float = 0.0,
    ) -> Tuple[np.ndarray, RobotState]:
        behavior_targets = np.zeros(Behavior.NUM_PARAMS)
        behavior_targets[Behavior.READY] = 1.0 if action_states[Action.ARMED] else 0.0

        current_yaw = float(sensors[State.Z_YAW])
        current_altitude = float(sensors[State.Z_ALTITUDE])
        current_yaw_unwrapped = self.search.unwrap_yaw(current_yaw)
        dt_estimate = self._dt(sim_time)

        if self.completed:
            self.target_height = float(np.clip(current_altitude, AUTO.MIN_HEIGHT, AUTO.MAX_HEIGHT))
            self.target_yaw = current_yaw
            self.target_thrust = 0.0
        else:
            within_blind_window = (sim_time - self.last_pass_time) < AUTO.POST_ADVANCE_BLIND_TIME
            expected_seen = self._tracking_matches_expected(tracking_result, within_blind_window)
            new_vision = self._is_new_vision(tracking_result)

            if new_vision:
                self.last_vision_timestamp = float(tracking_result.timestamp)

            if self.mode == "charge":
                if expected_seen and new_vision:
                    self._update_filtered_tracking(tracking_result)
                    self.last_detection_time = sim_time
                self._run_charge(current_yaw, current_altitude, sim_time, expected_seen)

            elif expected_seen:
                if new_vision or self.mode in ("search", "track"):
                    self._run_track(current_yaw, current_altitude, sim_time, tracking_result)

            else:
                self.charge_seen_counter = 0
                self.charge_candidate_started_time = -1e9

                if self.mode == "track" and (sim_time - self.last_detection_time) > AUTO.LOST_TIMEOUT:
                    self._enter_search_mode(current_yaw, current_yaw_unwrapped, sim_time, current_altitude)
                elif self.mode not in ("search", "charge"):
                    self._enter_search_mode(current_yaw, current_yaw_unwrapped, sim_time, current_altitude)

                if self.mode == "search":
                    cmd = self.search.step(
                        sim_time=sim_time,
                        current_yaw_wrapped=current_yaw,
                        current_yaw_unwrapped=current_yaw_unwrapped,
                        current_height=current_altitude,
                        nominal_height=self.search_nominal_height,
                        label=self.expected_label,
                        dt_estimate=dt_estimate,
                    )
                    self.target_yaw = cmd.target_yaw
                    self.target_height = cmd.target_height
                    self.target_thrust = cmd.target_thrust
                    self.debug_status = cmd.debug_status

        behavior_targets[Behavior.Z_HEIGHT] = self.target_height
        behavior_targets[Behavior.Z_YAW] = self.target_yaw
        behavior_targets[Behavior.FX_FORWARD] = self.target_thrust
        return behavior_targets, self