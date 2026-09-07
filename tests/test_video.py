import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from g1_llm_policy.video import VideoRecorder


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg tools required")
@pytest.mark.parametrize("speed", [1, 2])
def test_wait_duration_and_motion_frames(tmp_path, monkeypatch, speed):
    recorder = VideoRecorder(tmp_path, cameras=("overview",), fps=30, speed=speed)
    env = SimpleNamespace(data=SimpleNamespace(time=0.0))
    captions = []

    def render(*_):
        captions.append(recorder.caption)
        return np.zeros((32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(recorder, "render_overview", render)
    clock = iter([100.0, 102.0])
    monkeypatch.setattr("g1_llm_policy.video.time.monotonic", lambda: next(clock))
    try:
        recorder.capture(env)
        with recorder.waiting(env, None, 0):
            pass
        assert captions[-1] == "[0] Waiting for LLM"
        assert env.data.time == 0.0
        assert recorder.frames == 61
        # The two seconds of wait must not swallow subsequent simulation frames.
        for frame in range(1, 31):
            env.data.time = frame / 30
            recorder.capture(env)
        assert recorder.frames == 91
    finally:
        recorder.close()
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=nb_frames,duration",
        "-of", "json", str(tmp_path / "video-overview.mp4"),
    ])
    stream = json.loads(result)["streams"][0]
    assert int(stream["nb_frames"]) == 91
    assert float(stream["duration"]) == pytest.approx(91 / (30 * speed), abs=0.001)
