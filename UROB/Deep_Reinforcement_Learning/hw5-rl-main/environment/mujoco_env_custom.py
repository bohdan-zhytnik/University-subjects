from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
import glfw
from .mujoco_vecenv import MujocoEnv
import mujoco
import numpy as np
from typing import Union, Optional
from gymnasium.spaces import Space
from os import path
from threading import Lock

DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280


class extended_Viewer(WindowViewer):
    def __init__(self, model, data, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        super().__init__(model, data, width, height)
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._transparent = False
        self._contacts = False
        self._render_every_frame = True
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menu = False

        # Adjust for HiDPI displays
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width / window_width

        # Set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

    def render_to_array(self, cam_id=-1, depth=False):
        # Assume updated methods are correctly implemented as in mujoco 2.x
        rect = mujoco.MjrRect(0, 0, int(self._scale * self.viewport.width), int(self._scale * self.viewport.height))
        cam = mujoco.MjvCamera()
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.pert, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(rect, self.scn, self.con)

        # Reading pixels
        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)
        mujoco.mjr_readPixels(rgb_arr, depth_arr, rect, self.con)
        if depth:
            depth_img = depth_arr.reshape(rect.height, rect.width)
            return depth_img
        else:
            rgb_img = rgb_arr.reshape(rect.height, rect.width, 3)
            return rgb_img


class extendedEnv(MujocoEnv):
    def __init__(
        self,
        model: mujoco.MjModel,
        frame_skip: int,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self._initialize_simulation(model)
        self.frame_skip = frame_skip
        self.observation_space = observation_space
        self.render_mode = render_mode
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.viewer = None
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.metadata = {
            'render_modes': ['human', 'rgb_array', 'depth_array'],
            'render_fps': int(np.round(1.0 / self.dt))
        }

    def _initialize_simulation(self, model):
        if type(model) == str:
            if model.startswith("/"):
                fullpath = model
            elif model.startswith("./"):
                fullpath = model
            else:
                fullpath = path.join(path.dirname(__file__), "assets", model)
            if not path.exists(fullpath):
                raise OSError(f"File {fullpath} does not exist")
            self.model = mujoco.MjModel.from_xml_path(fullpath)
        else:
            self.model = model
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.model.vis.global_.fovy = 90
        self.data = mujoco.MjData(self.model)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        done = self._check_done()
        info = self._get_info()
        return obs, done, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = extended_Viewer(self.model, self.data, self.width, self.height)
                self.camera_setup()
            self.viewer.render()
        elif self.render_mode in ['rgb_array', 'depth_array']:
            if self.viewer is None:
                from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
                self.viewer = OffScreenViewer(self.model, self.data, self.width, self.height)
                self.camera_setup()
            return self.viewer.render_to_array(camera_id=self.camera_id, depth=(self.render_mode == 'depth_array'))
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        super().close()

    def camera_setup(self):
        assert self.viewer is not None, "Viewer must be initialized before setting up camera"
        # Implement according to the specifics of your environment
        pass

    def _get_obs(self):
        # Implement according to the specifics of your environment
        return self.data.qpos.ravel().copy()

    def _check_done(self):
        # Implement according to the specifics of your environment
        return False

    def _get_info(self):
        # Implement to provide additional data as per environment needs
        return {}

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
