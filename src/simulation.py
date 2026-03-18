import cv2
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

from .definitions import AXLE, THRUST_LEFT, THRUST_RIGHT, Action, State
from .vision.camera_config import CameraConfig
from .vision.image_processor import ImageProcessor
from .vision.target_tracker import TargetTracker


CAMERA_CONFIG = CameraConfig(
    resolution_name="HQVGA",
    fps=10.0,
    processing_mode="RGB",
    show_processed_window=True,
    window_scale=2,
)

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

        # Camera pipeline components
        self.camera_config = CAMERA_CONFIG
        self.image_processor = ImageProcessor(self.camera_config.processing_mode)
        self.target_tracker = TargetTracker()

        # Make sure the offscreen render target is large enough for selected camera size.
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), self.camera_config.width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), self.camera_config.height
        )

        # Trajectory recording
        self._assembly_id = self.model.body(ASSEMBLY).id
        self.traj_x: list[float] = []
        self.traj_y: list[float] = []
        self.traj_z: list[float] = []

        # Camera timing / cached frames
        self.camera_period = self.camera_config.period
        self.next_camera_capture_time = 0.0
        self.latest_camera_raw_rgb = None
        self.latest_camera_processed = None
        self.latest_camera_display_bgr = None
        self.latest_camera_capture_time = -1.0

        # Offscreen RGB readback buffer
        self.camera_rgb = np.empty(
            (self.camera_config.height, self.camera_config.width, 3), dtype=np.uint8
        )

        # OpenCV processed-view window state
        self.camera_window_name = self.camera_config.window_title
        self._camera_window_initialized = False

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
        self.cam.distance = 34.0
        self.cam.azimuth = 70
        self.cam.elevation = -18

        self.scene_main = mj.MjvScene(self.model, maxgeom=10000)
        self.scene_camera = mj.MjvScene(self.model, maxgeom=10000)

        self.context = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150.value
        )

        # Offscreen fixed camera
        self.camera_cam = mj.MjvCamera()
        self.camera_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.camera_cam.fixedcamid = self.model.camera(CAMERA).id

        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0.0
        self.lasty = 0.0

        glfw.set_key_callback(self.window, self._keyboard_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        mj.set_mjcb_control(self.controller.control_step)

        if self.camera_config.show_processed_window:
            self._init_processed_camera_window()

    def _keyboard_callback(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            self.camera_follow = True
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.next_camera_capture_time = self.data.time
            return

        self.controller.update_key_state(key, act)

    def _mouse_button_callback(self, window, button, act, mods):
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
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene_main,
            self.cam,
        )

    def _init_processed_camera_window(self):
        cv2.namedWindow(self.camera_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.camera_window_name,
            self.camera_config.width * self.camera_config.window_scale,
            self.camera_config.height * self.camera_config.window_scale,
        )
        self._camera_window_initialized = True

    def _get_hud_status(self):
        current_state = self.controller.state_machine.current_state
        tracking = self.controller.get_latest_tracking_result()
        sequence_index = getattr(current_state, "sequence_index", None)
        sequence = getattr(current_state, "GATE_SEQUENCE", None)
        gate_total = len(sequence) if sequence is not None else None
        gate_number = None
        if gate_total is not None and sequence_index is not None:
            gate_number = min(sequence_index + 1, gate_total)

        return {
            "state_name": type(current_state).__name__,
            "mode": getattr(current_state, "mode", "-"),
            "armed": bool(self.controller.action_states[Action.ARMED]),
            "active_target": getattr(current_state, "expected_label", None) or tracking.expected_label or "none",
            "gate_progress": "-" if gate_number is None or gate_total is None else f"{gate_number}/{gate_total}",
            "target_found": bool(tracking.found),
            "target_label": tracking.label or "none",
            "debug_status": getattr(current_state, "debug_status", "-"),
            "target_height": float(getattr(current_state, "target_height", 0.0)),
            "target_yaw": float(getattr(current_state, "target_yaw", 0.0)),
            "target_thrust": float(getattr(current_state, "target_thrust", 0.0)),
            "tracking": tracking,
        }

    def _annotate_display(self, display_bgr: np.ndarray, tracking_result) -> np.ndarray:
        annotated = display_bgr.copy()

        h, w = annotated.shape[:2]
        cv2.drawMarker(
            annotated,
            (w // 2, h // 2),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            18,
            1,
        )

        if tracking_result.found and tracking_result.center_x is not None and tracking_result.center_y is not None:
            cx = int(round(tracking_result.center_x))
            cy = int(round(tracking_result.center_y))
            bw = int(round(tracking_result.width_px or 0))
            bh = int(round(tracking_result.height_px or 0))
            x0 = max(cx - bw // 2, 0)
            y0 = max(cy - bh // 2, 0)
            x1 = min(cx + bw // 2, annotated.shape[1] - 1)
            y1 = min(cy + bh // 2, annotated.shape[0] - 1)

            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)

            label = tracking_result.label or "target"
            label_y = y0 - 8 if y0 >= 18 else y1 + 16
            cv2.putText(
                annotated,
                label,
                (x0, min(max(label_y, 14), annotated.shape[0] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def _show_processed_camera_window(self):
        if not self._camera_window_initialized or self.latest_camera_display_bgr is None:
            return
        cv2.imshow(self.camera_window_name, self.latest_camera_display_bgr)

    def _capture_and_process_camera_frame(self):
        glfw.make_context_current(self.window)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)

        viewport = mj.MjrRect(0, 0, self.camera_config.width, self.camera_config.height)

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.camera_cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene_camera,
        )
        mj.mjr_render(viewport, self.scene_camera, self.context)
        mj.mjr_readPixels(self.camera_rgb, None, viewport, self.context)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW.value, self.context)

        raw_rgb = np.flipud(self.camera_rgb).copy()
        processed_frame = self.image_processor.process(raw_rgb)
        tracking_expectation = self.controller.get_tracking_expectation()
        tracking_result = self.target_tracker.update(
            processed_frame.processed,
            self.data.time,
            expected_color=tracking_expectation.get("expected_color"),
            expected_shape=tracking_expectation.get("expected_shape"),
            expected_label=tracking_expectation.get("expected_label"),
        )
        display_bgr = self._annotate_display(
            processed_frame.display, tracking_result
        )

        self.latest_camera_raw_rgb = raw_rgb
        self.latest_camera_processed = processed_frame.processed
        self.latest_camera_display_bgr = display_bgr
        self.latest_camera_capture_time = float(self.data.time)

        self.controller.update_camera_frame(
            raw_rgb=raw_rgb,
            processed_frame=processed_frame.processed,
            display_bgr=display_bgr,
            sim_time=self.data.time,
            resolution_name=self.camera_config.resolution_name,
            processing_mode=self.camera_config.processing_mode,
            tracking_result=tracking_result,
        )

    def _render_main_window(self):
        glfw.make_context_current(self.window)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW.value, self.context)

        if self.camera_follow:
            pos = self.data.xpos[self._assembly_id]
            self.cam.lookat[:] = pos

        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        main_viewport = mj.MjrRect(0, 0, fb_width, fb_height)

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

        hud = self._get_hud_status()
        tracking = hud["tracking"]
        info_formatted = (
            f"{'ARMED' if hud['armed'] else 'DISARMED'}\n"
            f"{hud['state_name']}\n"
            f"{hud['mode']}\n"
            f"{hud['active_target']}\n"
            f"{hud['gate_progress']}\n"
            f"{'YES' if hud['target_found'] else 'NO'}\n"
            f"{hud['target_label']}\n"
            f"{hud['target_height']:8.3f}\n"
            f"{sensor_data[State.Z_ALTITUDE]:8.3f}\n"
            f"{hud['target_yaw']:8.3f}\n"
            f"{sensor_data[State.Z_YAW]:8.3f}\n"
            f"{hud['target_thrust']:8.3f}\n"
            f"{self.data.actuator(THRUST_LEFT).ctrl[0]:8.3f}\n"
            f"{self.data.actuator(THRUST_RIGHT).ctrl[0]:8.3f}\n"
            f"{self.data.joint(AXLE).qpos[0]:8.3f}\n"
            f"{self.latest_camera_capture_time:8.3f}\n"
            f"{self.camera_config.resolution_name}\n"
            f"{self.camera_config.processing_mode}\n"
            f"{tracking.debug_text}\n"
            f"{hud['debug_status']}"
        )
        info_labels = (
            "Arm State\n"
            "State\n"
            "Auto Mode\n"
            "Active Target\n"
            "Gate Progress\n"
            "Target Found\n"
            "Current Detection\n"
            "Altitude Target\n"
            "Altitude Actual\n"
            "Yaw Target\n"
            "Yaw Actual\n"
            "Target Thrust\n"
            "Motor L Thrust\n"
            "Motor R Thrust\n"
            "Servo Angle\n"
            "Camera Time\n"
            "Camera Res\n"
            "Proc Mode\n"
            "Tracker\n"
            "Status"
        )

        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            main_viewport,
            info_formatted,
            info_labels,
            self.context,
        )

        glfw.swap_buffers(self.window)

    def run(self):
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < 1.0 / 60.0:
                mj.mj_step(self.model, self.data)

            self._render_main_window()

            if self.data.time >= self.next_camera_capture_time:
                self._capture_and_process_camera_frame()
                self._show_processed_camera_window()

                self.next_camera_capture_time += self.camera_period
                while self.next_camera_capture_time <= self.data.time:
                    self.next_camera_capture_time += self.camera_period

            self._show_processed_camera_window()

            pos = self.data.xpos[self._assembly_id]
            self.traj_x.append(float(pos[0]))
            self.traj_y.append(float(pos[1]))
            self.traj_z.append(float(pos[2]))

            glfw.poll_events()

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if self._camera_window_initialized:
                visible = cv2.getWindowProperty(
                    self.camera_window_name, cv2.WND_PROP_VISIBLE
                )
                if visible < 1:
                    break

        self.stop()

    def stop(self):
        mj.set_mjcb_control(None)

        try:
            if len(self.traj_x) > 1:
                self._plot_3d_trajectory()
                self._plot_xy_trajectory()
        except Exception as e:
            print(f"Trajectory plotting skipped ({e})")

        cv2.destroyAllWindows()

        if getattr(self, "window", None) is not None:
            glfw.destroy_window(self.window)

        glfw.terminate()

    def _plot_3d_trajectory(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.traj_x, self.traj_y, self.traj_z, "-", linewidth=1.5, label="trajectory")
        ax.scatter(self.traj_x[0], self.traj_y[0], self.traj_z[0], c="green", s=40, label="start")
        ax.scatter(self.traj_x[-1], self.traj_y[-1], self.traj_z[-1], c="red", s=40, label="end")

        xs = np.array(self.traj_x)
        ys = np.array(self.traj_y)
        zs = np.array(self.traj_z)

        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        z_range = zs.max() - zs.min()
        max_range = max(x_range, y_range, z_range, 1e-9)

        ax.set_box_aspect((x_range / max_range, y_range / max_range, z_range / max_range))
        ax.set_title("Robot 3D trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def _plot_xy_trajectory(self):
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
