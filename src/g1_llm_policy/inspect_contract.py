"""G1 embodiment contract consumed directly by the pinned Inspect agent."""

import numpy as np
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Observation

from .commands import Command, Pose

CONTROL_HZ = 10.0
LABELS = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("x", "y", "z", "qw", "qx", "qy", "qz", "open")
)
SPACE = Box(
    (16,),
    low=np.array([-0.25, -0.85, 0.4, *([-1.0] * 4), 0.0] * 2),
    high=np.array([0.85, 0.85, 1.65, *([1.0] * 4), 1.0] * 2),
    semantics=ActionSemantics("eef_abs_pose", "quat_wxyz", "continuous", "world", LABELS),
)


def info(cameras=("ego",), hand="dex1"):
    obs = ObservationSpace(
        cameras=tuple(CameraSpec(c, 480, 640) for c in cameras),
        state=StateSpec((StateField("eef_state", (16,), "m+quat+normalized"),)),
    )
    return EmbodimentInfo(
        name="g1_mujoco",
        action_space=SPACE,
        observation_space=obs,
        control_hz=CONTROL_HZ,
        is_simulated=True,
        capabilities=frozenset({"resettable", "renderable"}),
        docs=(
            f"This embodiment is a SIMULATED fixed-base G1 with {hand} grippers. "
            "Both arms use world coordinates: +x forward, +y robot left, +z up, metres. "
            "Positions refer to the TCP itself (dex1: centre between jaws; dex3: palm). "
            "Do not add a wrist-to-TCP offset to these positions. "
            "Rotation is an absolute unit quaternion [qw,qx,qy,qz] mapping TCP to world. "
            "It is not an angle or a rotation relative to the initial pose. "
            "When changing a hand's orientation supply all four quaternion components; "
            "omit all four to preserve the measured orientation. Rotations use shortest-path SLERP. "
            "Local +x runs along "
            "the fingers; local y is the dex1 jaw opening direction. "
            "open is normalized: 0 closed, 1 open; observations use measured joint positions. "
            "Top-down grasping is prohibited. Do not approach the object from above "
            "with the fingers pointing downward or descend onto it to grasp. "
            "Use a side approach with the fingers approximately horizontal and grasp "
            "the object's sides. Lifting upward after a side grasp is allowed. "
            "Write notes and final explanations in English."
        ),
    )


def opening(state, side):
    joints = state["joints"]
    if state["gripper_model"] == "dex1":
        from .dex1 import CLOSED, OPEN

        q = np.mean([joints[f"{side}_dex1_finger_joint_{i}"] for i in (1, 2)])
        return float(np.clip((q - CLOSED) / (OPEN - CLOSED), 0, 1))
    return float(np.clip(1 - abs(joints[f"{side}_hand_index_0_joint"]) / 0.85, 0, 1))


def vector(state):
    values = []
    for side in ("left", "right"):
        pose = state["hands"][side]
        values.extend([*pose["position"], *pose["quaternion"], opening(state, side)])
    return np.array(values)


def observation(state, images, instruction, step=0):
    # Deliberate whitelist: no task truth, targets, tracking-error diagnostics or object state.
    return Observation(
        images=images,
        state={"eef_state": vector(state)},
        instruction=instruction,
        state_time=state["sim_time_s"],
        extra={"env_step": step, "approvals": state.get("approvals", [])},
    )


def decode(values, note="Inspect motion"):
    values = np.asarray(values, dtype=float)
    if values.shape != (16,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a finite 16-D Cartesian action")
    poses = {}
    for side, offset in (("left", 0), ("right", 8)):
        v = values[offset : offset + 8]
        poses[side] = Pose(position=tuple(v[:3]), quaternion=tuple(v[3:7]))
    return Command(status="move", **poses, left_open=values[7], right_open=values[15], note=note)


def pre_check(waypoints):
    try:
        for point in waypoints:
            decode(point)
    except ValueError as exc:
        return str(exc)
    return None
