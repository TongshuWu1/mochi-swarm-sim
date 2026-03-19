from dataclasses import dataclass

CAMERA_PRESETS = {
    "QQQVGA": (80, 60),
    "QQVGA": (160, 120),
    "HHQVGA": (120, 80),
    "HQVGA": (240, 160),
    "QVGA": (320, 240),
    "HVGA": (480, 320),
}

VALID_PROCESSING_MODES = {"RGB", "GRAY", "HSV"}


@dataclass(frozen=True)
class CameraConfig:
    resolution_name: str = "HQVGA"
    fps: float = 10.0
    processing_mode: str = "GRAY"
    show_processed_window: bool = True
    window_scale: int = 2

    @property
    def size(self) -> tuple[int, int]:
        if self.resolution_name not in CAMERA_PRESETS:
            raise ValueError(f"Unknown resolution '{self.resolution_name}'. Choose from: {list(CAMERA_PRESETS.keys())}")
        return CAMERA_PRESETS[self.resolution_name]

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def period(self) -> float:
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        return 1.0 / self.fps

    @property
    def normalized_processing_mode(self) -> str:
        mode = self.processing_mode.upper()
        if mode not in VALID_PROCESSING_MODES:
            raise ValueError(f"Unsupported processing mode '{self.processing_mode}'. Valid modes: {sorted(VALID_PROCESSING_MODES)}")
        return mode

    @property
    def window_title(self) -> str:
        return f"Nicla Vision Processed ({self.resolution_name}, {self.normalized_processing_mode})"
