"""Responsive GLFW viewer with a live ego inset, including on macOS/mjpython."""

import multiprocessing as mp
import time
import traceback

import glfw
import mujoco
import numpy as np

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def reset_camera(camera):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.12, 0, 0.68]
    camera.distance = 2.15
    camera.azimuth = 180
    camera.elevation = -35


def inset_rect(width, height):
    """Top right, always 4:3 like the images sent to the policy."""
    margin = max(4, round(min(width, height) * 0.018))
    w = max(4, int(min(width * 0.32, height * 0.42 * 4 / 3)) // 4 * 4)
    h = w * 3 // 4
    return mujoco.MjrRect(width - w - margin, height - h - margin, w, h)


class Window:
    """GLFW and OpenGL are owned exclusively by the child process's main thread."""

    def __init__(self, model):
        self.model = model
        self.context = None
        self.window = None
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the display")
        try:
            self.window = glfw.create_window(1100, 900, "G1 | Overview + Ego", None, None)
            if not self.window:
                raise RuntimeError("GLFW could not create the viewer window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.option = mujoco.MjvOption()
            self.camera = mujoco.MjvCamera()
            reset_camera(self.camera)
            self.ego = mujoco.MjvCamera()
            self.ego.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.ego.fixedcamid = model.camera("ego").id
            self.show_ego = True
            self.cursor = glfw.get_cursor_pos(self.window)
            glfw.set_cursor_pos_callback(self.window, self._mouse_move)
            glfw.set_mouse_button_callback(self.window, self._mouse_button)
            glfw.set_scroll_callback(self.window, self._scroll)
            glfw.set_key_callback(self.window, self._key)
        except BaseException:
            self.close()
            raise

    def _mouse_button(self, window, button, action, mods):
        self.cursor = glfw.get_cursor_pos(window)

    def _mouse_move(self, window, x, y):
        dx, dy = x - self.cursor[0], y - self.cursor[1]
        self.cursor = (x, y)
        height = max(1, glfw.get_window_size(window)[1])
        shift = any(
            glfw.get_key(window, key) == glfw.PRESS
            for key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        )
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.camera)

    def _scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y, self.camera)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            reset_camera(self.camera)
        elif key == glfw.KEY_E:
            self.show_ego = not self.show_ego

    def draw(self, data):
        width, height = glfw.get_framebuffer_size(self.window)
        if width < 16 or height < 16:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model, data, self.option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        if self.show_ego:
            inset = inset_rect(width, height)
            border = mujoco.MjrRect(
                inset.left - 2, inset.bottom - 2, inset.width + 4, inset.height + 4
            )
            mujoco.mjr_rectangle(border, 0.7, 0.75, 0.8, 1)
            mujoco.mjv_updateScene(
                self.model,
                data,
                self.option,
                None,
                self.ego,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(inset, self.scene, self.context)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                inset,
                "EGO",
                "",
                self.context,
            )
            # Keep the scene camera consistent with mouse operations on the main view.
            mujoco.mjv_updateCamera(self.model, data, self.camera, self.scene)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            "Drag: orbit | Right drag: pan | Scroll: zoom\n"
            "E: ego on/off | R: reset view | Esc: close",
            "",
            self.context,
        )

    def close(self):
        if self.context is not None:
            self.context.free()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()


def _run(model, state, stopping, startup, camera_state):
    window = None
    try:
        data = mujoco.MjData(model)
        window = Window(model)
        first = True
        while not stopping.is_set() and not glfw.window_should_close(window.window):
            started = time.monotonic()
            with state.get_lock():
                snapshot = np.frombuffer(state.get_obj()).copy()
            mujoco.mj_setState(model, data, snapshot, STATE)
            mujoco.mj_forward(model, data)
            with camera_state.get_lock():
                camera_state[:] = [
                    *window.camera.lookat,
                    window.camera.distance,
                    window.camera.azimuth,
                    window.camera.elevation,
                    *glfw.get_framebuffer_size(window.window),
                ]
            window.draw(data)
            glfw.swap_buffers(window.window)
            glfw.poll_events()
            if first:
                startup.send(None)
                first = False
            stopping.wait(max(0, 1 / 60 - (time.monotonic() - started)))
    except BaseException:
        error = traceback.format_exc()
        try:
            startup.send(error)
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        stopping.set()
        startup.close()
        if window:
            window.close()


class Viewer:
    def __init__(self, model, data):
        self.model, self.data = model, data
        # spawn gives Cocoa a real main thread even when the caller uses mjpython.
        ctx = mp.get_context("spawn")
        self.state = ctx.Array("d", mujoco.mj_stateSize(model, STATE))
        self.stopping = ctx.Event()
        self.camera_state = ctx.Array("d", [0.12, 0, 0.68, 2.15, 180, -35, 1100, 900])
        receiver, sender = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=_run,
            args=(model, self.state, self.stopping, sender, self.camera_state),
            name="g1-viewer",
            daemon=True,
        )
        self.sync()
        try:
            self.process.start()
            sender.close()
            if not receiver.poll(20):
                raise RuntimeError("Viewer did not open within 20 seconds")
            error = receiver.recv()
            if error:
                raise RuntimeError(f"Viewer could not start:\n{error}")
        except BaseException:
            self.close()
            raise
        finally:
            receiver.close()
            sender.close()

    def sync(self):
        with self.state.get_lock():
            mujoco.mj_getState(self.model, self.data, np.frombuffer(self.state.get_obj()), STATE)

    def camera_snapshot(self):
        with self.camera_state.get_lock():
            return list(self.camera_state[:])

    def is_running(self):
        return self.process.is_alive() and not self.stopping.is_set()

    def close(self):
        self.stopping.set()
        if self.process.pid is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
