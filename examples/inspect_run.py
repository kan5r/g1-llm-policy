"""Run the same G1/Codex loop through Inspect Robots's evaluation lifecycle."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspect_robots import eval
from inspect_robots.scene import Scene
from inspect_robots.task import Task

from g1_llm_policy.adapters.inspect_robots import G1Embodiment, InspectCodexPolicy
from g1_llm_policy.quaternion_tools import QuaternionApprover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--instruction",
        default="Move the right hand 3cm forward and 2cm up once from its initial pose, preserve orientation, then check the new observed pose and finish.",
    )
    args = parser.parse_args()
    embodiment = G1Embodiment()
    policy = InspectCodexPolicy(model=args.model)
    task = Task(
        name="g1-llm",
        scenes=[Scene(id="tabletop", instruction=args.instruction)],
        scorer=[],
        max_steps=args.max_steps,
    )
    try:
        logs = eval(
            task,
            policy,
            embodiment,
            log_dir="runs/inspect",
            store_frames=True,
            approver=QuaternionApprover(embodiment.info.action_space),
        )
        for log in logs:
            print(f"Inspect Robots: {log.status}")
            if log.status == "error":
                raise RuntimeError("Inspect Robots run failed; see runs/inspect")
    finally:
        policy.close()
        embodiment.close()


if __name__ == "__main__":
    main()
