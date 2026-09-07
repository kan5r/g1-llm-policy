"""Opt-in OpenGL check: G1_TEST_VIEWER=1 uv run pytest tests/test_viewer.py."""

import os

import glfw
import mujoco
import numpy as np
import pytest

from g1_llm_policy.env import G1Env
from g1_llm_policy.viewer import Window, inset_rect


@pytest.mark.skipif(os.environ.get("G1_TEST_VIEWER") != "1", reason="needs a desktop display")
def test_inset_matches_policy_camera_and_controls_do_not_change_physics():
    env = G1Env()
    window = Window(env.model)
    before = env.data.qpos.copy()
    try:
        for width, height in ((800, 600), (500, 700)):
            glfw.set_window_size(window.window, width, height)
            glfw.poll_events()
            window.draw(env.data)
            w, h = glfw.get_framebuffer_size(window.window)
            rgb = np.empty((h, w, 3), np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, w, h), window.context)
            rect = inset_rect(w, h)
            assert rect.width * 3 == rect.height * 4
            assert 0 <= rect.left < rect.left + rect.width <= w
            assert 0 <= rect.bottom < rect.bottom + rect.height <= h
            with mujoco.Renderer(env.model, height=rect.height, width=rect.width) as renderer:
                renderer.update_scene(env.data, camera="ego")
                expected = renderer.render().copy()
            glfw.make_context_current(window.window)
            actual = rgb[
                rect.bottom : rect.bottom + rect.height, rect.left : rect.left + rect.width
            ][::-1]
            # Exclude EGO label. Allow multisampling edge differences onscreen/offscreen.
            error = np.mean(np.abs(actual[60:].astype(float) - expected[60:].astype(float)))
            assert error < 1, error
        window._scroll(window.window, 0, 1)
        assert window.camera.distance < 2.15
        window._key(window.window, glfw.KEY_E, 0, glfw.PRESS, 0)
        assert not window.show_ego
        window.draw(env.data)
        window.camera.azimuth = 123
        window._key(window.window, glfw.KEY_R, 0, glfw.PRESS, 0)
        assert window.camera.azimuth == 180
        assert window.camera.distance == 2.15
        np.testing.assert_array_equal(env.data.qpos, before)
    finally:
        glfw.make_context_current(window.window)
        window.close()
        env.close()
