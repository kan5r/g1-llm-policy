"""Stream simulation-time frames to MP4 without accumulating images in memory."""

import shutil
import subprocess
from contextlib import ExitStack

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VideoRecorder:
    def __init__(self, directory, cameras=("overview", "ego"), fps=30, speed=2):
        self.directory = directory
        self.cameras = cameras
        self.fps = fps
        self.speed = speed
        self.start = None
        self.renderer = None
        self.frames = 0
        self.caption = "Ready"
        self.last_frames = []
        self.writers = []
        self.resources = ExitStack()
        self.executable = shutil.which("ffmpeg")
        if not self.executable:
            raise RuntimeError("Video requires ffmpeg: brew install ffmpeg (or use --no-video)")

    def set_command(self, step, command):
        self.caption = f"[{step}] {command.status}: {command.note}"

    def render_overview(self, env, viewer):
        snapshot = viewer.camera_snapshot() if viewer else [0.12, 0, 0.68, 2.15, 180, -35]
        camera = mujoco.MjvCamera()
        camera.lookat[:] = snapshot[:3]
        camera.distance, camera.azimuth, camera.elevation = snapshot[3:6]
        # A fixed 4:3 scene: window resizing must not change the recorded composition.
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.model, width=960, height=720)
        self.renderer.update_scene(env.data, camera=camera)
        canvas = Image.new("RGB", (960, 864), (18, 21, 27))
        canvas.paste(Image.fromarray(self.renderer.render()), (0, 0))
        ego = Image.fromarray(env.render("ego")).resize((288, 216), Image.Resampling.LANCZOS)
        canvas.paste(ego, (656, 16))
        draw = ImageDraw.Draw(canvas)
        speed_font = ImageFont.load_default(size=24)
        speed_label = f"{self.speed:g}x"
        badge_width = int(draw.textlength(speed_label, font=speed_font)) + 24
        draw.rounded_rectangle((16, 16, 16 + badge_width, 56), radius=6, fill=(18, 21, 27))
        draw.text((28, 23), speed_label, font=speed_font, fill="white")
        draw.rectangle((654, 14, 945, 233), outline=(200, 210, 220), width=2)
        draw.rectangle((656, 16, 704, 39), fill=(18, 21, 27))
        draw.text((662, 18), "EGO", font=ImageFont.load_default(size=18), fill="white")
        # Fit the complete note into the caption band, including long policy outputs.
        for size in range(22, 5, -1):
            font = ImageFont.load_default(size=size)
            lines, line = [], ""
            for word in self.caption.split():
                candidate = f"{line} {word}" if line else word
                if line and draw.textlength(candidate, font=font) > 920:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            if len(lines) * (size + 5) <= 120:
                break
        draw.multiline_text((20, 732), "\n".join(lines), font=font, fill="white", spacing=5)
        return np.asarray(canvas)

    def capture(self, env, viewer=None, force=False):
        if self.start is None:
            self.start = env.data.time
        if not force and env.data.time - self.start + 1e-8 < self.frames / self.fps:
            return
        self.last_frames = []
        for index, camera in enumerate(self.cameras):
            frame = (
                self.render_overview(env, viewer) if camera == "overview" else env.render(camera)
            )
            if len(self.writers) <= index:
                height, width = frame.shape[:2]
                stderr = self.resources.enter_context(
                    (self.directory / f"video-{camera}.log").open("wb")
                )
                writer = subprocess.Popen(
                    [
                        self.executable,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        f"{width}x{height}",
                        "-framerate",
                        str(self.fps * self.speed),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(self.directory / f"video-{camera}.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                    stderr=stderr,
                )
                self.writers.append(writer)
            payload = frame.tobytes()
            self.writers[index].stdin.write(payload)
            self.last_frames.append(payload)
        self.frames += 1

    def hold_final(self, env, viewer=None):
        self.capture(env, viewer, force=True)
        # 1.5 seconds of playback to read the final action, independent of speed.
        for _ in range(round(self.fps * self.speed * 1.5)):
            for writer, payload in zip(self.writers, self.last_frames):
                writer.stdin.write(payload)

    def close(self):
        failed = False
        try:
            for writer in self.writers:
                try:
                    writer.stdin.close()
                except BrokenPipeError:
                    failed = True
            for writer in self.writers:
                failed |= writer.wait() != 0
        finally:
            self.resources.close()
            if self.renderer:
                self.renderer.close()
                self.renderer = None
        if failed:
            raise RuntimeError(f"Video encoding failed; see {self.directory}/video-*.log")
