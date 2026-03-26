from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

import cv2
import numpy as np

from ..preferences import TURBULENCE as TURB


@dataclass
class LocalWind:
    force_xyz: np.ndarray
    torque_xyz: np.ndarray
    base_force_xy: np.ndarray
    angle_deg: float
    base_angle_deg: float
    scale: float
    scale_target: float
    angle_target_deg: float


class TurbulenceField:
    def __init__(self):
        self.enabled = bool(TURB.ENABLED_DEFAULT)
        self.show_hud = bool(TURB.SHOW_HUD_DEFAULT)
        self.show_field_window = bool(TURB.SHOW_FIELD_WINDOW_DEFAULT)
        self.window_name = TURB.FIELD_WINDOW_NAME
        self.window_size = int(TURB.FIELD_WINDOW_SIZE)

        self.scale = 1.0
        self.scale_target = 1.0
        self.angle_offset_rad = 0.0
        self.angle_target_rad = 0.0
        self._next_resample_time = 0.0
        self._rng = random.Random(7)
        self._window_initialized = False

        self._sample_new_targets(0.0, force_now=True)

    def reset(self, sim_time: float):
        self.scale = 1.0
        self.scale_target = 1.0
        self.angle_offset_rad = 0.0
        self.angle_target_rad = 0.0
        self._next_resample_time = sim_time
        self._sample_new_targets(sim_time, force_now=True)

    def toggle_enabled(self):
        self.enabled = not self.enabled

    def toggle_hud(self):
        self.show_hud = not self.show_hud

    def toggle_field_window(self):
        self.show_field_window = not self.show_field_window
        if not self.show_field_window and self._window_initialized:
            cv2.destroyWindow(self.window_name)
            self._window_initialized = False

    def _sample_new_targets(self, sim_time: float, force_now: bool = False):
        self.scale_target = self._rng.uniform(TURB.SCALE_MIN, TURB.SCALE_MAX)
        self.angle_target_rad = math.radians(
            self._rng.uniform(TURB.ANGLE_MIN_DEG, TURB.ANGLE_MAX_DEG)
        )
        dt = self._rng.uniform(TURB.TARGET_RESAMPLE_MIN_S, TURB.TARGET_RESAMPLE_MAX_S)
        self._next_resample_time = sim_time + dt
        if force_now:
            self.scale = self.scale_target
            self.angle_offset_rad = self.angle_target_rad

    def update(self, sim_time: float, dt: float):
        if sim_time >= self._next_resample_time:
            self._sample_new_targets(sim_time)
        tau = max(float(TURB.DRIFT_TIME_CONSTANT_S), 1e-6)
        alpha = 1.0 - math.exp(-max(dt, 0.0) / tau)
        self.scale += alpha * (self.scale_target - self.scale)
        self.angle_offset_rad += alpha * (self.angle_target_rad - self.angle_offset_rad)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _normalized_coords(self, x: float, y: float) -> tuple[float, float]:
        nx = 2.0 * (x - TURB.FIELD_X_MIN) / max(TURB.FIELD_X_MAX - TURB.FIELD_X_MIN, 1e-6) - 1.0
        ny = 2.0 * (y - TURB.FIELD_Y_MIN) / max(TURB.FIELD_Y_MAX - TURB.FIELD_Y_MIN, 1e-6) - 1.0
        return float(np.clip(nx, -1.5, 1.5)), float(np.clip(ny, -1.5, 1.5))

    def base_field_at(self, x: float, y: float) -> tuple[np.ndarray, float, float, float]:
        nx, ny = self._normalized_coords(x, y)

        # Smooth spatial pattern with swirl + waves.
        vx = 0.75 * math.cos(1.35 * ny + 0.35 * math.sin(1.1 * nx)) + 0.45 * ny
        vy = 0.85 * math.sin(1.25 * nx - 0.30 * math.cos(1.0 * ny)) - 0.35 * nx
        v = np.array([vx, vy], dtype=float)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            v = np.array([1.0, 0.0], dtype=float)
            norm = 1.0
        direction = v / norm

        pattern = 0.5 * (math.sin(1.7 * nx) * math.cos(1.25 * ny) + 1.0)
        mag = self._lerp(TURB.XY_FORCE_MIN, TURB.XY_FORCE_MAX, pattern)
        zpattern = math.sin(1.1 * nx + 0.7 * ny)
        fz = self._lerp(TURB.Z_FORCE_MIN, TURB.Z_FORCE_MAX, 0.5 * (zpattern + 1.0))
        yz = math.cos(1.35 * nx - 0.8 * ny)
        tz = self._lerp(TURB.YAW_TORQUE_MIN, TURB.YAW_TORQUE_MAX, 0.5 * (yz + 1.0))
        return direction * mag, mag, fz, tz

    def local_wind(self, pos_xyz: Iterable[float]) -> LocalWind:
        x, y, _z = [float(v) for v in pos_xyz]
        base_xy, _mag, fz, tz = self.base_field_at(x, y)
        c = math.cos(self.angle_offset_rad)
        s = math.sin(self.angle_offset_rad)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        actual_xy = self.scale * (rot @ base_xy)
        force_xyz = np.array([actual_xy[0], actual_xy[1], self.scale * fz], dtype=float)
        torque_xyz = np.array([0.0, 0.0, self.scale * tz], dtype=float)
        ang = math.degrees(math.atan2(actual_xy[1], actual_xy[0]))
        base_ang = math.degrees(math.atan2(base_xy[1], base_xy[0]))
        return LocalWind(
            force_xyz=force_xyz,
            torque_xyz=torque_xyz,
            base_force_xy=base_xy,
            angle_deg=ang,
            base_angle_deg=base_ang,
            scale=float(self.scale),
            scale_target=float(self.scale_target),
            angle_target_deg=math.degrees(self.angle_target_rad),
        )

    def apply_to_data(self, data, body_id: int):
        data.xfrc_applied[body_id, :] = 0.0
        if not self.enabled:
            return self.local_wind(data.xpos[body_id])
        local = self.local_wind(data.xpos[body_id])
        data.xfrc_applied[body_id, 0:3] = local.force_xyz
        data.xfrc_applied[body_id, 3:6] = local.torque_xyz
        return local

    def _ensure_window(self):
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size, self.window_size)
        self._window_initialized = True

    def render_field_window(self, blimp_pos_xyz: Iterable[float]):
        if not self.show_field_window:
            if self._window_initialized:
                cv2.destroyWindow(self.window_name)
                self._window_initialized = False
            return
        self._ensure_window()

        size = self.window_size
        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        # Grid arrows of base field
        xs = np.linspace(TURB.FIELD_X_MIN, TURB.FIELD_X_MAX, TURB.FIELD_GRID_NX)
        ys = np.linspace(TURB.FIELD_Y_MIN, TURB.FIELD_Y_MAX, TURB.FIELD_GRID_NY)
        for x in xs:
            for y in ys:
                base_xy, _, _, _ = self.base_field_at(float(x), float(y))
                self._draw_arrow(canvas, x, y, base_xy, (80, 140, 240), scale=180.0)

        # Robot current actual local wind
        px, py, _ = [float(v) for v in blimp_pos_xyz]
        local = self.local_wind((px, py, 0.0))
        self._draw_arrow(canvas, px, py, local.force_xyz[:2], (20, 170, 20), scale=260.0, thickness=3)
        u, v = self._world_to_image(px, py, size)
        cv2.circle(canvas, (u, v), 7, (20, 170, 20), -1)

        # border and legends
        cv2.rectangle(canvas, (8, 8), (size - 8, size - 8), (60, 60, 60), 1)
        cv2.putText(canvas, 'Base field: orange', (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 140, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Robot/local wind: green', (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 170, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, f'Scale {self.scale:.2f}->{self.scale_target:.2f}', (18, size - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,40,40), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Angle {math.degrees(self.angle_offset_rad):+.1f} deg -> {math.degrees(self.angle_target_rad):+.1f} deg', (18, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,40,40), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, canvas)

    def _world_to_image(self, x: float, y: float, size: int) -> tuple[int, int]:
        tx = (x - TURB.FIELD_X_MIN) / max(TURB.FIELD_X_MAX - TURB.FIELD_X_MIN, 1e-6)
        ty = (y - TURB.FIELD_Y_MIN) / max(TURB.FIELD_Y_MAX - TURB.FIELD_Y_MIN, 1e-6)
        u = int(np.clip(tx, 0.0, 1.0) * (size - 1))
        v = int((1.0 - np.clip(ty, 0.0, 1.0)) * (size - 1))
        return u, v

    def _draw_arrow(self, canvas: np.ndarray, x: float, y: float, vec_xy: np.ndarray, color: tuple[int, int, int], scale: float, thickness: int = 2):
        size = canvas.shape[0]
        u, v = self._world_to_image(x, y, size)
        dx = int(round(float(vec_xy[0]) * scale))
        dy = int(round(float(vec_xy[1]) * scale))
        end = (u + dx, v - dy)
        cv2.arrowedLine(canvas, (u, v), end, color, thickness, cv2.LINE_AA, tipLength=0.25)
