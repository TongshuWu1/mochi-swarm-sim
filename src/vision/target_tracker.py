from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackingResult:
    found: bool = False
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    width_px: Optional[float] = None
    height_px: Optional[float] = None
    area_px: Optional[float] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    offset_x_norm: Optional[float] = None
    offset_y_norm: Optional[float] = None
    color_name: Optional[str] = None
    shape_name: Optional[str] = None
    label: Optional[str] = None
    expected_color: Optional[str] = None
    expected_shape: Optional[str] = None
    expected_label: Optional[str] = None
    matched_expected: bool = False
    timestamp: float = -1.0
    debug_text: str = "no target"
    score: float = 0.0
    confidence: float = 0.0


class TargetTracker:
    """Detect colored balloons with temporal smoothing.

    Uses HSV color gating plus blob/circle checks so the blue skybox is rejected.
    """

    COLOR_RANGES = {
        "red": [((0, 110, 70), (10, 255, 255)), ((170, 110, 70), (180, 255, 255))],
        "orange": [((11, 105, 85), (20, 255, 255))],
        "yellow": [((21, 100, 95), (35, 255, 255))],
        "lime": [((36, 85, 70), (52, 255, 255))],
        "green": [((53, 75, 55), (84, 255, 255))],
        "cyan": [((85, 70, 70), (102, 255, 255))],
        "blue": [((103, 80, 60), (126, 255, 255))],
        "purple": [((127, 75, 60), (145, 255, 255))],
        "magenta": [((146, 85, 70), (169, 255, 255))],
    }

    def __init__(self):
        self.latest_result = TrackingResult()
        self._ema = None
        self._ema_alpha = 0.38
        self._max_roi_expand = 2.8
        self._ema_label = None
        self._roi_label = None

        # Shape filters tuned for the simulated balloons.
        self._min_area_px = 55.0
        self._min_blob_dim_px = 8
        self._max_area_frac = 0.22
        self._min_solidity = 0.78
        self._min_fill_ratio = 0.50
        self._min_aspect = 0.62
        self._min_circularity = 0.60
        self._min_circle_extent = 0.58
        self._min_confidence = 0.64

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        expected_color: Optional[str] = None,
        expected_shape: Optional[str] = None,
        expected_label: Optional[str] = None,
    ) -> TrackingResult:
        h, w = frame.shape[:2]
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.latest_result = TrackingResult(
                found=False,
                frame_width=w,
                frame_height=h,
                timestamp=float(timestamp),
                expected_color=expected_color,
                expected_shape=expected_shape,
                expected_label=expected_label,
                debug_text="tracker expects RGB frame",
            )
            return self.latest_result

        if expected_label != self._roi_label:
            self._ema = None
            self._ema_label = None
            self._roi_label = expected_label

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        best = None

        x0, y0, x1, y1 = self._build_roi(w, h, expected_label)
        roi_slice = np.s_[y0:y1, x0:x1]

        for color_name, ranges in self.COLOR_RANGES.items():
            if expected_color is not None and color_name != expected_color:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask_roi = cv2.inRange(
                    hsv[roi_slice],
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                )
                mask[roi_slice] |= mask_roi

            kernel_open = np.ones((3, 3), np.uint8)
            kernel_close = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                candidate = self._score_contour(contour, color_name, w, h)
                if candidate is None:
                    continue
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

        if best is None:
            self._ema = None
            self._ema_label = None
            message = "no balloon detected"
            if expected_label:
                message = f"no {expected_label} balloon detected"
            self.latest_result = TrackingResult(
                found=False,
                frame_width=w,
                frame_height=h,
                timestamp=float(timestamp),
                expected_color=expected_color,
                expected_shape=expected_shape,
                expected_label=expected_label,
                matched_expected=False,
                debug_text=message,
            )
            return self.latest_result

        best = self._smooth_best(best, expected_label)
        label = f"{best['color_name']} balloon"
        matched_expected = (expected_color is None) or (best["color_name"] == expected_color)
        dx = (best["cx"] - 0.5 * w) / max(0.5 * w, 1.0)
        dy = (best["cy"] - 0.5 * h) / max(0.5 * h, 1.0)
        self.latest_result = TrackingResult(
            found=True,
            center_x=best["cx"],
            center_y=best["cy"],
            width_px=best["bw"],
            height_px=best["bh"],
            area_px=best["area"],
            frame_width=w,
            frame_height=h,
            offset_x_norm=dx,
            offset_y_norm=dy,
            color_name=best["color_name"],
            shape_name="balloon",
            label=label,
            expected_color=expected_color,
            expected_shape=expected_shape,
            expected_label=expected_label,
            matched_expected=matched_expected,
            timestamp=float(timestamp),
            score=float(best["score"]),
            confidence=float(best["confidence"]),
            debug_text=(
                f"{label} | conf={best['confidence']:.2f} | area={best['area']:.0f}px | "
                f"dx={dx:+.2f} | dy={dy:+.2f}"
            ),
        )
        return self.latest_result

    def _score_contour(self, contour, color_name: str, frame_w: int, frame_h: int):
        area = float(cv2.contourArea(contour))
        if area < self._min_area_px or area > self._max_area_frac * frame_w * frame_h:
            return None

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < self._min_blob_dim_px or bh < self._min_blob_dim_px:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            return None

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1e-6)
        fill_ratio = area / max(float(bw * bh), 1e-6)
        aspect = min(bw, bh) / max(max(bw, bh), 1e-6)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-9)

        (circle_x, circle_y), circle_r = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * max(circle_r, 1e-6) * max(circle_r, 1e-6)
        circle_extent = area / max(circle_area, 1e-6)

        if (
            solidity < self._min_solidity
            or fill_ratio < self._min_fill_ratio
            or aspect < self._min_aspect
            or circularity < self._min_circularity
            or circle_extent < self._min_circle_extent
        ):
            return None

        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-9:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = x + bw / 2.0
            cy = y + bh / 2.0

        center_bias = self._center_bias(cx, cy, frame_w, frame_h)
        circle_center_error = np.hypot(cx - circle_x, cy - circle_y) / max(circle_r, 1.0)
        center_consistency = float(np.clip(1.0 - 0.6 * circle_center_error, 0.0, 1.0))

        confidence = float(np.clip(
            0.32 * circularity
            + 0.20 * aspect
            + 0.18 * solidity
            + 0.18 * circle_extent
            + 0.12 * center_consistency,
            0.0,
            1.0,
        ))
        if confidence < self._min_confidence:
            return None

        score = area * (0.55 + 0.45 * confidence) * (0.72 + 0.28 * center_bias)
        return {
            "score": float(score),
            "confidence": confidence,
            "color_name": color_name,
            "shape_name": "balloon",
            "cx": float(cx),
            "cy": float(cy),
            "bw": float(bw),
            "bh": float(bh),
            "area": float(area),
        }

    def _build_roi(self, w: int, h: int, expected_label: Optional[str]):
        if self._ema is None or expected_label is None or expected_label != self._ema_label:
            return 0, 0, w, h

        cx = self._ema["cx"]
        cy = self._ema["cy"]
        bw = max(56.0, self._ema["bw"] * self._max_roi_expand)
        bh = max(56.0, self._ema["bh"] * self._max_roi_expand)
        x0 = int(np.clip(cx - 0.5 * bw, 0, w - 1))
        y0 = int(np.clip(cy - 0.5 * bh, 0, h - 1))
        x1 = int(np.clip(cx + 0.5 * bw, x0 + 1, w))
        y1 = int(np.clip(cy + 0.5 * bh, y0 + 1, h))
        return x0, y0, x1, y1

    def _smooth_best(self, best: dict, expected_label: Optional[str]) -> dict:
        if self._ema is None or self._ema_label != expected_label or self.latest_result.color_name != best["color_name"]:
            self._ema = dict(best)
            self._ema_label = expected_label
            return dict(best)

        a = self._ema_alpha
        for key in ("cx", "cy", "bw", "bh", "area", "score", "confidence"):
            self._ema[key] = (1.0 - a) * self._ema[key] + a * best[key]
        self._ema["color_name"] = best["color_name"]
        self._ema["shape_name"] = best["shape_name"]
        return dict(self._ema)

    @staticmethod
    def _center_bias(cx: float, cy: float, w: int, h: int) -> float:
        dx = (cx - 0.5 * w) / max(0.5 * w, 1.0)
        dy = (cy - 0.5 * h) / max(0.5 * h, 1.0)
        return float(np.clip(1.0 - 0.42 * np.hypot(dx, dy), 0.2, 1.0))
