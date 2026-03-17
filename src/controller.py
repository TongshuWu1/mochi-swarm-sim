import numpy as np
from mujoco.glfw import glfw
from scipy.spatial.transform import Rotation as R

from .state.robot_state_machine import RobotStateMachine
from .state.manual_state import ManualState
from .state.exploration_state import ExplorationState
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


KEY_BINDINGS = {
    glfw.KEY_W: Action.FORWARD,
    glfw.KEY_S: Action.BACKWARD,
    glfw.KEY_A: Action.LEFT,
    glfw.KEY_D: Action.RIGHT,
    glfw.KEY_SPACE: Action.UP,
    glfw.KEY_LEFT_SHIFT: Action.DOWN,
    glfw.KEY_ENTER: Action.ARMED,
}


class Controller:
    """
    Handles flight logic, sensor parsing, and camera-frame access.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.action_states = {action: False for action in Action}
        self.state_machine = RobotStateMachine(ManualState())
        self.robot = Differential()
        self.senses = np.zeros(State.NUM_STATES)

        # Latest camera frame from Simulation
        # Expected shape: (320, 480, 3), dtype=uint8, RGB
        self.latest_camera_rgb = None
        self.latest_camera_time = -1.0
        self.latest_camera_seq = -1
        self._last_consumed_camera_seq = -1

    def update_key_state(self, key, action):
        """
        Called by the simulation keyboard callback.
        """
        is_pressed = action != glfw.RELEASE

        # State selection by number keys
        if action == glfw.PRESS:
            if key == glfw.KEY_1:
                self.state_machine.current_state = ManualState()
                print("[STATE SELECT] ManualState (1)")
                return

            if key == glfw.KEY_2:
                self.state_machine.current_state = ExplorationState()
                print("[STATE SELECT] ExplorationState (2)")
                return

        if key in KEY_BINDINGS:
            mapped_action = KEY_BINDINGS[key]

            if mapped_action == Action.ARMED:
                # Toggle only on press
                if action == glfw.PRESS:
                    self.action_states[Action.ARMED] = not self.action_states[
                        Action.ARMED
                    ]
                    print(
                        f"[ARMED] {'ON' if self.action_states[Action.ARMED] else 'OFF'}"
                    )
            else:
                # Hold behavior
                self.action_states[mapped_action] = is_pressed

    def update_camera_frame(self, frame_rgb, sim_time):
        """
        Called by Simulation whenever the camera window refreshes.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), RGB uint8
            sim_time: current MuJoCo simulation time
        """
        self.latest_camera_rgb = frame_rgb
        self.latest_camera_time = float(sim_time)
        self.latest_camera_seq += 1

    def has_camera_frame(self) -> bool:
        """
        True once at least one camera frame has been received.
        """
        return self.latest_camera_rgb is not None

    def has_new_camera_frame(self) -> bool:
        """
        True if a newer frame has arrived since the last consume call.
        """
        return self.latest_camera_seq > self._last_consumed_camera_seq

    def get_latest_camera_rgb(self, copy: bool = False):
        """
        Returns the latest camera RGB frame, or None if unavailable.

        Args:
            copy: if True, returns a copy
        """
        if self.latest_camera_rgb is None:
            return None
        return self.latest_camera_rgb.copy() if copy else self.latest_camera_rgb

    def consume_latest_camera_rgb(self, copy: bool = False):
        """
        Returns the latest RGB frame and marks it as consumed.
        Useful when you want to process each new frame only once.
        """
        if self.latest_camera_rgb is None:
            return None
        self._last_consumed_camera_seq = self.latest_camera_seq
        return self.latest_camera_rgb.copy() if copy else self.latest_camera_rgb

    def get_latest_camera_info(self):
        """
        Returns metadata about the latest camera frame.
        """
        return {
            "available": self.latest_camera_rgb is not None,
            "time": self.latest_camera_time,
            "seq": self.latest_camera_seq,
            "is_new": self.has_new_camera_frame(),
            "shape": None
            if self.latest_camera_rgb is None
            else self.latest_camera_rgb.shape,
        }

    def control_step(self, model, data):
        """
        Main MuJoCo control callback.
        """

        # --- Sense ---
        self._sense()

        # --- Optional future vision hook ---
        # Example:
        # if self.has_new_camera_frame():
        #     frame_rgb = self.consume_latest_camera_rgb(copy=False)
        #     # Do image processing here

        # --- State machine ---
        behavior_commands = self.state_machine.update(self.senses, self.action_states)

        # --- Low-level controller ---
        actuator_commands = self.robot.control(self.senses, behavior_commands)

        # --- Apply actuator commands ---
        data.actuator(THRUST_LEFT).ctrl = actuator_commands[0]
        data.actuator(THRUST_RIGHT).ctrl = actuator_commands[1]
        data.actuator(SERVO).ctrl = actuator_commands[2]

    def _sense(self):
        """
        State vector:
        [z_altitude, z_altitude_vel, x_roll, y_pitch, z_yaw,
         x_roll_rate, y_pitch_rate, z_yaw_rate]
        """

        # Position sensor: [x, y, z]
        imu_pos = self.data.sensor(IMU_POS).data.copy()
        self.senses[State.Z_ALTITUDE] = imu_pos[2]

        # Linear velocity sensor: [vx, vy, vz]
        imu_lin_vel = self.data.sensor(IMU_LIN_VEL).data.copy()
        self.senses[State.Z_ALTITUDE_VEL] = imu_lin_vel[2]

        # Quaternion from MuJoCo is [w, x, y, z]
        quat = self.data.sensor(IMU_QUAT).data.copy()
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [x,y,z,w]
        roll, pitch, yaw = rot.as_euler("xyz", degrees=False)

        self.senses[State.X_ROLL] = roll
        self.senses[State.Y_PITCH] = pitch
        self.senses[State.Z_YAW] = yaw

        # Angular velocity sensor: [wx, wy, wz]
        ang_vel = self.data.sensor(IMU_ANG_VEL).data.copy()
        self.senses[State.X_ROLL_RATE] = ang_vel[0]
        self.senses[State.Y_PITCH_RATE] = ang_vel[1]
        self.senses[State.Z_YAW_RATE] = ang_vel[2]