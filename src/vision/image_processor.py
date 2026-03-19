from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ProcessedFrame:
    raw_rgb: np.ndarray
    processed: np.ndarray
    display: np.ndarray
    mode: str


class ImageProcessor:
    """Simple pluggable image-processing stage."""

    def __init__(self, mode: str = "RGB"):
        self.mode = mode.upper()

    def process(self, raw_rgb: np.ndarray) -> ProcessedFrame:
        if self.mode == "RGB":
            display = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            return ProcessedFrame(raw_rgb=raw_rgb, processed=raw_rgb.copy(), display=display, mode=self.mode)

        if self.mode == "GRAY":
            gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return ProcessedFrame(raw_rgb=raw_rgb, processed=gray, display=display, mode=self.mode)

        if self.mode == "HSV":
            hsv = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2HSV)
            display = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            return ProcessedFrame(raw_rgb=raw_rgb, processed=hsv, display=display, mode=self.mode)

        raise ValueError(f"Unsupported processing mode: {self.mode}")
