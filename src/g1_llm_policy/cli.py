"""Run with mjpython on macOS when opening the native viewer."""

import argparse
import json
import math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from PIL import Image

from .commands import Command, Pose
from .env import G1Env
from .policies.codex import AppServer
from .policies.inspect_agent import InspectPolicy as CodexPolicy
from .verification import resolve_task
from .video import VideoRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["models", "view", "demo", "run"])
    parser.add_argument(
        "--instruction",
        default="pick up the red block",
    )
    parser.add_argument("--task", choices=["auto", "pickup", "none"], default="auto")
    parser.add_argument("--hand", choices=["dex3", "dex1"], default="dex1")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--images", choices=["always", "on_demand"], default="always")
    parser.add_argument("--max-speed-frac", type=float, default=0.1)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--cameras", nargs="+", default=["ego"], choices=["ego", "overview", "front"]
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-speed", type=float, default=2.0)
    parser.add_argument(
        "--video-cameras",
        nargs="+",
        default=["overview", "ego"],
        choices=["ego", "overview", "front"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.video_speed) or args.video_speed <= 0:
        parser.error("--video-speed must be a finite positive number")
    if not 1 <= args.video_fps <= 100:
        parser.error("--video-fps must be between 1 and 100")
    if args.mode == "models":
        with AppServer(timeout=args.timeout) as server:
            for model in server.models():
                print(model["model"])
        return
    out = args.output or Path("runs") / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True, exist_ok=True)
    env = G1Env(hand=args.hand)
    recorder = None
    policy = None
    try:
        if not args.no_video and args.mode in ("run", "demo"):
            recorder = VideoRecorder(
                out, tuple(dict.fromkeys(args.video_cameras)), args.video_fps, args.video_speed
            )
        viewer_context = nullcontext(None)
        if args.viewer or args.mode == "view":
            from .viewer import Viewer

            viewer_context = Viewer(env.model, env.data)
        with viewer_context as viewer:
            if recorder:
                recorder.capture(env, viewer)

            def sync(_):
                if recorder:
                    recorder.capture(env, viewer)
                if viewer:
                    if not viewer.is_running():
                        raise KeyboardInterrupt
                    viewer.sync()

            if args.mode == "view":
                while viewer.is_running():
                    env.move(Command.hold(), duration=0.1, callback=sync, realtime=True)
                return
            Image.fromarray(env.render()).save(out / "initial.png")
            if args.mode == "demo":
                pose = env.pose("right")
                command = Command.hold("Scripted right-hand reach +4cm x, +2cm z")
                command.right = Pose(
                    position=(pose.position[0] + 0.04, pose.position[1], pose.position[2] + 0.02),
                    quaternion=pose.quaternion,
                )
                if recorder:
                    recorder.set_command(0, command)
                env.move(command, callback=sync, realtime=bool(viewer))
                Image.fromarray(env.render()).save(out / "final.png")
                (out / "result.json").write_text(json.dumps(env.observe(), indent=2))
                print(json.dumps(env.report, indent=2))
                if not env.report["right"]["reached"]:
                    raise RuntimeError("Scripted reach did not converge")
            else:
                with AppServer(timeout=args.timeout, log_path=out / "app-server.jsonl") as server:
                    policy = CodexPolicy(
                        server,
                        args.model,
                        args.effort,
                        hand=args.hand,
                        cameras=args.cameras,
                        max_llm_calls=args.max_llm_calls,
                        images=args.images,
                        max_speed_frac=args.max_speed_frac,
                    )
                    with (out / "episode.jsonl").open("w") as log:
                        for step in range(args.max_steps):
                            images = {camera: env.render(camera) for camera in args.cameras}
                            for camera, image in images.items():
                                Image.fromarray(image).save(out / f"{step:03d}-{camera}.png")
                            before = env.observe()
                            command = policy.act(args.instruction, before, images)
                            print(f"[{step}] {command.status}: {command.note}", flush=True)
                            if recorder:
                                recorder.set_command(step, command)
                            record = {
                                "step": step,
                                "instruction": args.instruction,
                                "model": args.model,
                                "before": before,
                                "command": command.model_dump(),
                            }
                            decision = env.execute(
                                command,
                                task=resolve_task(args.task, args.instruction),
                                callback=sync,
                                realtime=bool(viewer),
                            )
                            record["waypoints"] = [c.model_dump() for c in command._waypoints]
                            (out / "agent-transcript.json").write_text(
                                json.dumps(policy.transcript(), indent=2)
                            )
                            record["decision"] = decision
                            record["physics"] = env.pickup.report()
                            if recorder and command.status == "done":
                                recorder.caption += (
                                    " | Physics: " + decision["verification"]["reason"]
                                    if decision.get("verification")
                                    else ""
                                )
                            if command.status == "done" and decision["outcome"] == "failed":
                                print(
                                    f"[{step}] evaluation FAILED: {decision['verification']['reason']}",
                                    flush=True,
                                )
                            record["after"] = env.observe()
                            log.write(json.dumps(record) + "\n")
                            log.flush()
                            if decision["stop"]:
                                break
                        else:
                            print("Stopped at --max-steps; completion is not established.")
                Image.fromarray(env.render()).save(out / "final.png")
            if recorder:
                recorder.hold_final(env, viewer)
            print(f"Output: {out.resolve()}")
    finally:
        try:
            if recorder:
                recorder.close()
        finally:
            if policy:
                policy.close()
            env.close()


if __name__ == "__main__":
    main()
