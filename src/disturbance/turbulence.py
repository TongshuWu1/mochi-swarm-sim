from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass
class LocalWindSample:
    force_world: np.ndarray
    yaw_torque: float
    base_force_world: np.ndarray
    base_heading_deg: float
    base_magnitude: float
    actual_heading_deg: float
    actual_magnitude: float
    scale: float
    angle_offset_deg: float


class StaticFieldTurbulence:
    """Spatial wind field with slow global random drift in angle/magnitude.

    Each point has a fixed base direction/magnitude from the spatial field. Over time, the
    whole field gets a slowly drifting magnitude scale and angle offset. This keeps the map
    visually stable while still feeling a bit more alive and realistic.
    """

    def __init__(self, prefs):
        self.p = prefs
        self.enabled = bool(self.p.ENABLED_BY_DEFAULT)
        self.hud_visible = True
        self.graph_visible = True

        self._rng = np.random.default_rng(self.p.SEED)
        self._scale = 1.0
        self._scale_target = 1.0
        self._angle_offset = 0.0
        self._angle_target = 0.0
        self._next_target_time = 0.0
        self._last_update_time: float | None = None

        self._x_min = self.p.FIELD_X_MIN
        self._x_max = self.p.FIELD_X_MAX
        self._y_min = self.p.FIELD_Y_MIN
        self._y_max = self.p.FIELD_Y_MAX

    def reset(self, sim_time: float = 0.0) -> None:
        self._scale = 1.0
        self._scale_target = 1.0
        self._angle_offset = 0.0
        self._angle_target = 0.0
        self._next_target_time = sim_time
        self._last_update_time = sim_time

    def toggle_enabled(self) -> None:
        self.enabled = not self.enabled

    def toggle_hud(self) -> None:
        self.hud_visible = not self.hud_visible

    def toggle_graph(self) -> None:
        self.graph_visible = not self.graph_visible

    def update(self, sim_time: float) -> None:
        if self._last_update_time is None:
            self.reset(sim_time)
        dt = max(0.0, sim_time - float(self._last_update_time))
        self._last_update_time = sim_time

        if sim_time >= self._next_target_time:
            self._scale_target = float(
                self._rng.uniform(self.p.SCALE_MIN, self.p.SCALE_MAX)
            )
            self._angle_target = math.radians(
                float(self._rng.uniform(-self.p.MAX_ANGLE_OFFSET_DEG, self.p.MAX_ANGLE_OFFSET_DEG))
            )
            self._next_target_time = sim_time + float(
                self._rng.uniform(self.p.TARGET_HOLD_TIME_MIN, self.p.TARGET_HOLD_TIME_MAX)
            )

        alpha = 1.0 - math.exp(-dt / max(self.p.SMOOTHING_TIME_CONSTANT, 1e-6))
        self._scale += alpha * (self._scale_target - self._scale)
        self._angle_offset += alpha * (self._angle_target - self._angle_offset)

    def sample(self, position_world: Iterable[float]) -> LocalWindSample:
        pos = np.asarray(list(position_world), dtype=float)
        base_force = self._base_force_world(pos[0], pos[1], pos[2] if pos.size > 2 else 0.0)
        base_mag_xy = float(np.linalg.norm(base_force[:2]))
        base_heading = math.degrees(math.atan2(base_force[1], base_force[0])) if base_mag_xy > 1e-9 else 0.0

        actual_force = base_force.copy()
        rotated_xy = self._rotate_xy(actual_force[:2], self._angle_offset)
        actual_force[:2] = rotated_xy * self._scale
        actual_force[2] *= self._scale

        actual_mag_xy = float(np.linalg.norm(actual_force[:2]))
        actual_heading = math.degrees(math.atan2(actual_force[1], actual_force[0])) if actual_mag_xy > 1e-9 else 0.0
        yaw_torque = float(np.clip(self.p.YAW_TORQUE_GAIN * actual_force[1], -self.p.YAW_TORQUE_MAX, self.p.YAW_TORQUE_MAX))

        return LocalWindSample(
            force_world=actual_force,
            yaw_torque=yaw_torque,
            base_force_world=base_force,
            base_heading_deg=base_heading,
            base_magnitude=base_mag_xy,
            actual_heading_deg=actual_heading,
            actual_magnitude=actual_mag_xy,
            scale=float(self._scale),
            angle_offset_deg=math.degrees(self._angle_offset),
        )

    def build_field_image(
        self,
        width: int,
        height: int,
        blimp_pos: np.ndarray | None = None,
        balloon_points_xy: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        img = np.full((height, width, 3), 20, dtype=np.uint8)
        grid_cols = self.p.FIELD_GRID_COLS
        grid_rows = self.p.FIELD_GRID_ROWS

        for r in range(grid_rows):
            for c in range(grid_cols):
                x = self._x_min + (c + 0.5) * (self._x_max - self._x_min) / grid_cols
                y = self._y_min + (r + 0.5) * (self._y_max - self._y_min) / grid_rows
                sample = self.sample((x, y, 0.0))
                px, py = self._world_to_canvas(x, y, width, height)
                self._draw_arrow(img, (px, py), sample.base_force_world[:2], self.p.FIELD_ARROW_SCALE, (90, 160, 255))

        # border
        cv = __import__('cv2')
        cv.rectangle(img, (8, 8), (width - 9, height - 9), (80, 80, 80), 1)
        cv.putText(img, 'Base field (orange) + blimp local actual wind (green)', (14, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv.LINE_AA)

        if balloon_points_xy:
            for bx, by in balloon_points_xy:
                px, py = self._world_to_canvas(bx, by, width, height)
                cv.circle(img, (px, py), 3, (0, 165, 255), -1)

        if blimp_pos is not None:
            px, py = self._world_to_canvas(float(blimp_pos[0]), float(blimp_pos[1]), width, height)
            sample = self.sample(blimp_pos)
            cv.circle(img, (px, py), 5, (0, 255, 0), -1)
            self._draw_arrow(img, (px, py), sample.force_world[:2], self.p.FIELD_ARROW_SCALE * 1.35, (0, 255, 0))
            cv.putText(img, f"local |Fxy|={sample.actual_magnitude:.3f}", (14, height - 18), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv.LINE_AA)

        return img

    def _base_force_world(self, x: float, y: float, z: float) -> np.ndarray:
        xr = self._normalize(x, self._x_min, self._x_max)
        yr = self._normalize(y, self._y_min, self._y_max)

        theta = (
            self.p.BASE_HEADING_DEG
            + self.p.X_HEADING_VARIATION_DEG * math.sin(2.0 * math.pi * xr)
            + self.p.Y_HEADING_VARIATION_DEG * math.cos(2.0 * math.pi * yr)
        )
        theta_rad = math.radians(theta)

        mag01 = (
            0.50
            + 0.26 * math.sin(2.0 * math.pi * xr + 0.45)
            + 0.18 * math.cos(2.0 * math.pi * yr - 0.35)
            + 0.10 * math.sin(2.0 * math.pi * (xr + 0.7 * yr))
        )
        mag01 = float(np.clip(mag01, 0.0, 1.0))
        mag_xy = self.p.HORIZONTAL_MAG_MIN + mag01 * (self.p.HORIZONTAL_MAG_MAX - self.p.HORIZONTAL_MAG_MIN)

        fz = self.p.VERTICAL_MAG_MAX * (0.35 * math.sin(2.0 * math.pi * xr - 0.2) - 0.25 * math.cos(2.0 * math.pi * yr + 0.5))
        fz = float(np.clip(fz, -self.p.VERTICAL_MAG_MAX, self.p.VERTICAL_MAG_MAX))

        return np.array([
            mag_xy * math.cos(theta_rad),
            mag_xy * math.sin(theta_rad),
            fz,
        ], dtype=float)

    @staticmethod
    def _rotate_xy(v: np.ndarray, angle_rad: float) -> np.ndarray:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)

    @staticmethod
    def _normalize(v: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.5
        return (v - lo) / (hi - lo)

    def _world_to_canvas(self, x: float, y: float, width: int, height: int) -> tuple[int, int]:
        px = int(round(12 + (x - self._x_min) / max(self._x_max - self._x_min, 1e-9) * (width - 24)))
        py = int(round(height - 12 - (y - self._y_min) / max(self._y_max - self._y_min, 1e-9) * (height - 24)))
        return px, py

    @staticmethod
    def _draw_arrow(img: np.ndarray, start: tuple[int, int], vec_xy: np.ndarray, scale: float, color: tuple[int, int, int]) -> None:
        cv = __import__('cv2')
        vx, vy = float(vec_xy[0]), float(vec_xy[1])
        mag = math.hypot(vx, vy)
        if mag < 1e-8:
            return
        end = (int(round(start[0] + vx / mag * mag * scale)), int(round(start[1] - vy / mag * mag * scale)))
        cv.arrowedLine(img, start, end, color, 1, cv.LINE_AA, tipLength=0.25)
