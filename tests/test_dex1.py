import hashlib
import json

import numpy as np
import pytest

from g1_llm_policy.commands import Command, Pose
from g1_llm_policy.dex1 import ASSETS, CLOSED, OPEN, TCP_X
from g1_llm_policy.env import G1Env


def test_official_assets_unchanged():
    source = json.loads((ASSETS / "SOURCE.json").read_text())
    for path, digest in source["sha256"].items():
        assert hashlib.sha256((ASSETS / path).read_bytes()).hexdigest() == digest


def test_dex1_model_and_independent_hands():
    env = G1Env(hand="dex1")
    try:
        assert env.model.nu == 18
        assert env.model.neq == 2
        for side in ("left", "right"):
            np.testing.assert_allclose(env.model.site(f"{side}_palm").pos, [TCP_X, 0, 0])
            assert env.model.body(f"{side}_wrist_yaw_link").pos[0] == pytest.approx(0.051)
            np.testing.assert_allclose(
                env.model.jnt_actfrcrange[env.model.joint(f"{side}_wrist_yaw_joint").id],
                [-13.4, 13.4],
            )
        env.move(Command.hold().model_copy(update={"right_open": 0}))
        for side, expected in (("left", OPEN), ("right", CLOSED)):
            values = [env.data.joint(f"{side}_dex1_finger_joint_{i}").qpos[0] for i in (1, 2)]
            np.testing.assert_allclose(values, expected, atol=2e-4)
    finally:
        env.close()


def test_dex1_box_grasp_lift_and_release():
    # Place a box within easy reach on the unchanged table. This tests the
    # physical gripper, not autonomous planning from the default scene.
    env = G1Env(hand="dex1", object_shape="box")
    try:
        target = np.array(env.pose("right").position)
        table = env.model.geom("table")
        target[2] = table.pos[2] + table.size[2] + 0.025
        adr = env.model.jnt_qposadr[env.model.body("cube").jntadr[0]]
        env.model.qpos0[adr : adr + 3] = target
        env.reset()

        def move(position, opening):
            command = Command.hold()
            command.right = Pose(position=tuple(position), quaternion=(1, 0, 0, 0))
            command.right_open = opening
            env.move(command)

        move(target, 1)
        move(target, 0)
        assert not env.pickup.report()["success"]
        move(target + [0, 0, 0.07], 0)
        report = env.pickup.report()
        assert report["success"]
        assert report["contact_regions"]["right"] == ["jaw_1", "jaw_2"]
        move(target + [0, 0, 0.07], 1)
        assert not env.pickup.report()["success"]
        assert env.pickup.report()["clearance_m"] < 0.002
    finally:
        env.close()
