import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

from .definitions import AXLE, THRUST_LEFT, THRUST_RIGHT, Action, State


# Camera view size: QVGA
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_MARGIN = 20

# Simulated camera refresh rate
CAMERA_FPS = 12.0
CAMERA_PERIOD = 1.0 / CAMERA_FPS

CAMERA = "nicla_vision"
ASSEMBLY = "assembly"


class Simulation:
    def __init__(self, model, data, controller):
        self.model = model
        self.data = data
        self.controller = controller

        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.camera_follow = True

        # Trajectory recording
        self._assembly_id = self.model.body(ASSEMBLY).id
        self.traj_x = []
        self.traj_y = []
        self.traj_z = []

        # Camera timing / cached frame
        self.camera_period = CAMERA_PERIOD
        self.next_camera_capture_time = 0.0
        self.latest_camera_rgb = None
        self.latest_camera_capture_time = -1.0

        # CPU buffer for camera-pane readback
        self.camera_rgb = np.empty((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        # --- GLFW init ---
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        if mode is None:
            glfw.terminate()
            raise RuntimeError("Failed to get primary monitor video mode")

        self.window = glfw.create_window(
            mode.size.width,
            mode.size.height,
            "Mochi Simulation",
            None,
            None,
        )
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create main GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.cam.distance = 20.0
        self.cam.azimuth = 60
        self.cam.elevation = -20

        self.scene_main = mj.MjvScene(self.model, maxgeom=10000)
        self.scene_camera = mj.MjvScene(self.model, maxgeom=10000)

        self.context = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150.value
        )

        # Fixed onboard camera for the rendered camera pane
        self.camera_cam = mj.MjvCamera()
        self.camera_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.camera_cam.fixedcamid = self.model.camera(CAMERA).id

        # --- Input state ---
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0.0
        self.lasty = 0.0

        # --- Callbacks ---
        glfw.set_key_callback(self.window, self._keyboard_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        # MuJoCo control callback
        mj.set_mjcb_control(self.controller.control_step)

    # -------------------------------------------------------------------------
    # Input callbacks
    # -------------------------------------------------------------------------

    def _keyboard_callback(self, window, key, scancode, act, mods):
        """Handles keyboard input."""
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            self.camera_follow = True
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            return

        self.controller.update_key_state(key, act)

    def _mouse_button_callback(self, window, button, act, mods):
        """Handles mouse button input for camera control."""
        self.button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self.lastx, self.lasty = glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window, xpos, ypos):
        """Handles mouse movement for camera control."""
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx, self.lasty = xpos, ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        _, height = glfw.get_window_size(window)
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self.button_right:
            self.camera_follow = False
            action = (
                mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self.button_left:
            action = (
                mj.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mj.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            self.model,
            action,
            dx / max(height, 1),
            dy / max(height, 1),
            self.scene_main,
            self.cam,
        )

    def _scroll_callback(self, window, xoffset, yoffset):
        """Handles mouse scroll for camera zoom."""
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene_main,
            self.cam,
        )

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _camera_viewport_rect(self, fb_width, fb_height):
        """
        Place the QVGA camera pane in the bottom-right corner.
        Coordinates are in framebuffer pixels.
        """
        x = fb_width - CAMERA_WIDTH - CAMERA_MARGIN
        y = CAMERA_MARGIN
        return mj.MjrRect(x, y, CAMERA_WIDTH, CAMERA_HEIGHT)

    def _render_main_window(self):
        """Render the main simulation window."""
        glfw.make_context_current(self.window)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW.value, self.context)

        if self.camera_follow:
            pos = self.data.xpos[self._assembly_id]
            self.cam.lookat[:] = pos

        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        main_viewport = mj.MjrRect(0, 0, fb_width, fb_height)

        # Main world view
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene_main,
        )
        mj.mjr_render(main_viewport, self.scene_main, self.context)

        # Sensor overlay
        sensor_data = self.controller.senses
        labels = [name.lower() for name in State.__members__ if name != "NUM_STATES"]
        sensors_formatted = "\n".join(f"{val:8.3f}" for val in sensor_data)
        sensors_labels = "\n".join(labels)

        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            main_viewport,
            sensors_formatted,
            sensors_labels,
            self.context,
        )

        # Status overlay
        armed = "ARMED" if self.controller.action_states[Action.ARMED] else "DISARMED"
        info_formatted = (
            f"{armed}\n"
            f"{self.controller.state_machine.current_state.target_height:8.3f}\n"
            f"{sensor_data[State.Z_ALTITUDE]:8.3f}\n"
            f"{self.controller.state_machine.current_state.target_yaw:8.3f}\n"
            f"{sensor_data[State.Z_YAW]:8.3f}\n"
            f"{self.controller.state_machine.current_state.target_thrust:8.3f}\n"
            f"{self.data.actuator(THRUST_LEFT).ctrl[0]:8.3f}\n"
            f"{self.data.actuator(THRUST_RIGHT).ctrl[0]:8.3f}\n"
            f"{self.data.joint(AXLE).qpos[0]:8.3f}\n"
            f"{self.latest_camera_capture_time:8.3f}"
        )
        info_labels = (
            "Status\n"
            "Altitude Target\n"
            "Altitude Actual\n"
            "Yaw Target\n"
            "Yaw Actual\n"
            "Target Thrust\n"
            "Motor L Thrust\n"
            "Motor R Thrust\n"
            "Servo Angle\n"
            "Camera Time"
        )

        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            main_viewport,
            info_formatted,
            info_labels,
            self.context,
        )

        # Draw camera pane only when a new camera frame is scheduled.
        # This keeps the visual refresh closer to the simulated camera rate.
        if self.data.time >= self.next_camera_capture_time:
            camera_viewport = self._camera_viewport_rect(fb_width, fb_height)

            mj.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.camera_cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scene_camera,
            )
            mj.mjr_render(camera_viewport, self.scene_camera, self.context)

            # Read back the exact QVGA camera pane for image processing
            mj.mjr_readPixels(self.camera_rgb, None, camera_viewport, self.context)

            frame = np.flipud(self.camera_rgb).copy()
            self.latest_camera_rgb = frame
            self.latest_camera_capture_time = float(self.data.time)

            # Push frame into controller for image processing
            self.controller.update_camera_frame(frame, self.data.time)

            self.next_camera_capture_time += self.camera_period
            while self.next_camera_capture_time <= self.data.time:
                self.next_camera_capture_time += self.camera_period

        glfw.swap_buffers(self.window)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def run(self):
        """Starts the main simulation loop."""
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < 1.0 / 60.0:
                mj.mj_step(self.model, self.data)

            self._render_main_window()

            # Record trajectory once per rendered frame
            pos = self.data.xpos[self._assembly_id]
            self.traj_x.append(float(pos[0]))
            self.traj_y.append(float(pos[1]))
            self.traj_z.append(float(pos[2]))

            glfw.poll_events()

        self.stop()

    # -------------------------------------------------------------------------
    # Shutdown / plots
    # -------------------------------------------------------------------------

    def stop(self):
        mj.set_mjcb_control(None)

        try:
            if len(self.traj_x) > 1:
                self._plot_3d_trajectory()
                self._plot_xy_trajectory()
        except Exception as e:
            print(f"Trajectory plotting skipped ({e})")

        if getattr(self, "window", None) is not None:
            glfw.destroy_window(self.window)

        glfw.terminate()

    def _plot_3d_trajectory(self):
        """Render a 3D plot of the recorded trajectory."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.traj_x,
            self.traj_y,
            self.traj_z,
            "-",
            linewidth=1.5,
            label="trajectory",
        )
        ax.scatter(
            self.traj_x[0],
            self.traj_y[0],
            self.traj_z[0],
            c="green",
            s=40,
            label="start",
        )
        ax.scatter(
            self.traj_x[-1],
            self.traj_y[-1],
            self.traj_z[-1],
            c="red",
            s=40,
            label="end",
        )

        xs = np.array(self.traj_x)
        ys = np.array(self.traj_y)
        zs = np.array(self.traj_z)

        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        z_range = zs.max() - zs.min()
        max_range = max(x_range, y_range, z_range, 1e-9)

        ax.set_box_aspect(
            (x_range / max_range, y_range / max_range, z_range / max_range)
        )

        ax.set_title("Robot 3D trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def _plot_xy_trajectory(self):
        """Render a 2D top-down plot of the recorded trajectory."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax2 = fig.add_subplot(111)
        ax2.plot(self.traj_x, self.traj_y, "-", linewidth=1.5, label="trajectory")
        ax2.scatter(self.traj_x[0], self.traj_y[0], c="green", s=40, label="start")
        ax2.scatter(self.traj_x[-1], self.traj_y[-1], c="red", s=40, label="end")
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_title("Robot XY trajectory")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="best")
        plt.tight_layout()
        plt.show()