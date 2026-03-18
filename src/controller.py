import numpy as np
from mujoco.glfw import glfw
from scipy.spatial.transform import Rotation as R

from .state.robot_state_machine import RobotStateMachine
from .state.manual_state import ManualState
from .state.exploration_state import ExplorationState
from .state.auto_gate_sequence_state import AutoGateSequenceState
from .robot.differential import Differential
from .definitions import (
    Action,
    State,
    SERVO,
    THRUST_RIGHT,
    THRUST_LEFT,
    IMU_POS,
    IMU_LIN_VEL,
    IMU_ANG_VEL,
    IMU_QUAT,
)
from .vision.target_tracker import TrackingResult


KEY_BINDINGS = {
    glfw.KEY_W: Action.FORWARD,
    glfw.KEY_S: Action.BACKWARD,
    glfw.KEY_A: Action.LEFT,
    glfw.KEY_D: Action.RIGHT,
    glfw.KEY_SPACE: Action.UP,
    glfw.KEY_LEFT_SHIFT: Action.DOWN,
    glfw.KEY_ENTER: Action.ARMED,
    glfw.KEY_3: Action.AUTO_MODE,
}


class Controller:
    """Handles flight logic, sensor parsing, and camera/tracking data access."""

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.action_states = {action: False for action in Action}
        self.state_machine = RobotStateMachine(ManualState())
        self.robot = Differential()
        self.senses = np.zeros(State.NUM_STATES)

        # Camera frames from Simulation
        self.latest_camera_raw_rgb = None
        self.latest_camera_processed = None
        self.latest_camera_display_bgr = None
        self.latest_camera_time = -1.0
        self.latest_camera_seq = -1
        self._last_consumed_camera_seq = -1
        self.latest_camera_resolution_name = None
        self.latest_camera_processing_mode = None

        # Tracking output from the visual pipeline
        self.latest_tracking_result = TrackingResult()

    def update_key_state(self, key, action):
        is_pressed = action != glfw.RELEASE

        if action == glfw.PRESS:
            if key == glfw.KEY_1:
                self.state_machine.current_state = ManualState()
                print("[STATE SELECT] ManualState (1)")
                return
            if key == glfw.KEY_2:
                self.state_machine.current_state = ExplorationState()
                print("[STATE SELECT] ExplorationState (2)")
                return
            if key == glfw.KEY_3:
                self.state_machine.current_state = AutoGateSequenceState()
                print("[STATE SELECT] AutoGateSequenceState (3)")
                return

        if key in KEY_BINDINGS:
            mapped_action = KEY_BINDINGS[key]
            if mapped_action == Action.ARMED:
                if action == glfw.PRESS:
                    self.action_states[Action.ARMED] = not self.action_states[Action.ARMED]
                    print(f"[ARMED] {'ON' if self.action_states[Action.ARMED] else 'OFF'}")
            else:
                self.action_states[mapped_action] = is_pressed

    def update_camera_frame(
        self,
        raw_rgb,
        processed_frame,
        display_bgr,
        sim_time,
        resolution_name=None,
        processing_mode=None,
        tracking_result: TrackingResult | None = None,
    ):
        self.latest_camera_raw_rgb = raw_rgb
        self.latest_camera_processed = processed_frame
        self.latest_camera_display_bgr = display_bgr
        self.latest_camera_time = float(sim_time)
        self.latest_camera_seq += 1
        self.latest_camera_resolution_name = resolution_name
        self.latest_camera_processing_mode = processing_mode
        if tracking_result is not None:
            self.latest_tracking_result = tracking_result

    def has_camera_frame(self) -> bool:
        return self.latest_camera_processed is not None

    def has_new_camera_frame(self) -> bool:
        return self.latest_camera_seq > self._last_consumed_camera_seq

    def get_latest_camera_processed(self, copy: bool = False):
        if self.latest_camera_processed is None:
            return None
        return self.latest_camera_processed.copy() if copy else self.latest_camera_processed

    def get_latest_camera_raw_rgb(self, copy: bool = False):
        if self.latest_camera_raw_rgb is None:
            return None
        return self.latest_camera_raw_rgb.copy() if copy else self.latest_camera_raw_rgb

    def get_latest_camera_display_bgr(self, copy: bool = False):
        if self.latest_camera_display_bgr is None:
            return None
        return self.latest_camera_display_bgr.copy() if copy else self.latest_camera_display_bgr

    def consume_latest_camera_processed(self, copy: bool = False):
        if self.latest_camera_processed is None:
            return None
        self._last_consumed_camera_seq = self.latest_camera_seq
        return self.latest_camera_processed.copy() if copy else self.latest_camera_processed

    def get_latest_tracking_result(self) -> TrackingResult:
        return self.latest_tracking_result

    def get_tracking_expectation(self):
        state = self.state_machine.current_state
        return {
            "expected_color": getattr(state, "expected_color", None),
            "expected_shape": getattr(state, "expected_shape", None),
            "expected_label": getattr(state, "expected_label", None),
            "state_name": type(state).__name__,
        }

    def get_latest_camera_info(self):
        return {
            "available": self.latest_camera_processed is not None,
            "time": self.latest_camera_time,
            "seq": self.latest_camera_seq,
            "is_new": self.has_new_camera_frame(),
            "raw_shape": None if self.latest_camera_raw_rgb is None else self.latest_camera_raw_rgb.shape,
            "processed_shape": None if self.latest_camera_processed is None else self.latest_camera_processed.shape,
            "display_shape": None if self.latest_camera_display_bgr is None else self.latest_camera_display_bgr.shape,
            "resolution_name": self.latest_camera_resolution_name,
            "processing_mode": self.latest_camera_processing_mode,
            "tracking": self.latest_tracking_result,
        }

    def control_step(self, model, data):
        self._sense()

        # Future hook: if you want to drive from vision, use:
        # if self.has_new_camera_frame():
        #     frame = self.consume_latest_camera_processed(copy=False)
        #     tracking = self.get_latest_tracking_result()

        behavior_commands = self.state_machine.update(
            self.senses,
            self.action_states,
            tracking_result=self.latest_tracking_result,
            sim_time=float(data.time),
        )
        actuator_commands = self.robot.control(self.senses, behavior_commands)

        data.actuator(THRUST_LEFT).ctrl = actuator_commands[0]
        data.actuator(THRUST_RIGHT).ctrl = actuator_commands[1]
        data.actuator(SERVO).ctrl = actuator_commands[2]

    def _sense(self):
        imu_pos = self.data.sensor(IMU_POS).data.copy()
        self.senses[State.Z_ALTITUDE] = imu_pos[2]

        imu_lin_vel = self.data.sensor(IMU_LIN_VEL).data.copy()
        self.senses[State.Z_ALTITUDE_VEL] = imu_lin_vel[2]

        quat = self.data.sensor(IMU_QUAT).data.copy()
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = rot.as_euler("xyz", degrees=False)

        self.senses[State.X_ROLL] = roll
        self.senses[State.Y_PITCH] = pitch
        self.senses[State.Z_YAW] = yaw

        ang_vel = self.data.sensor(IMU_ANG_VEL).data.copy()
        self.senses[State.X_ROLL_RATE] = ang_vel[0]
        self.senses[State.Y_PITCH_RATE] = ang_vel[1]
        self.senses[State.Z_YAW_RATE] = ang_vel[2]
