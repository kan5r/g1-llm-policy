import hashlib
import json

import mujoco
import numpy as np
import pytest

from g1_llm_policy.commands import Command, Pose, output_schema
from g1_llm_policy.env import G1Env
from g1_llm_policy.model import ASSETS


@pytest.fixture
def env():
    instance = G1Env(hand="dex3")
    yield instance
    instance.close()


def test_asset_provenance():
    source = json.loads((ASSETS / "SOURCE.json").read_text())
    for path, sha in source["sha256"].items():
        assert hashlib.sha256((ASSETS / path).read_bytes()).hexdigest() == sha


def test_model_fixed_base_black_hands_ego(env):
    assert env.model.nu == 28
    assert env.model.nv == 34  # 28 actuated DOFs + the free object's six DOFs.
    assert env.model.cam_bodyid[env.model.camera("ego").id] == env.model.body("torso_link").id
    black = env.model.mat("black").id
    count = 0
    for i in range(env.model.ngeom):
        mesh = env.model.geom_dataid[i]
        if env.model.geom_group[i] == 2 and mesh >= 0:
            name = env.model.mesh(mesh).name
            if "hand_" in name:
                assert env.model.geom_matid[i] == black
                count += 1
    assert count >= 16


@pytest.mark.parametrize("side", ["left", "right"])
def test_physical_reach_and_hold_other_hand(env, side):
    initial = env.pose(side)
    other = "right" if side == "left" else "left"
    held = np.array(env.pose(other).position)
    command = Command.hold()
    setattr(
        command,
        side,
        Pose(
            position=(initial.position[0] + 0.04, initial.position[1], initial.position[2] + 0.02),
            quaternion=initial.quaternion,
        ),
    )
    start_time = env.data.time
    env.move(command)
    assert env.data.time - start_time == pytest.approx(2.0)
    assert env.report[side]["position_error_m"] < 0.015
    assert env.report[side]["orientation_error_rad"] < 0.05
    assert np.linalg.norm(np.array(env.pose(other).position) - held) < 0.01


def test_fingers_move_physical_object_stays_dynamic(env):
    before = env.data.qpos.copy()
    command = Command.hold()
    command.right_open = 0
    env.move(command)
    joint = env.model.joint("right_hand_index_0_joint")
    assert env.data.qpos[joint.qposadr[0]] > before[joint.qposadr[0]] + 0.3
    # Gravity/contact is active; cube settles on the table instead of being attached.
    table = env.model.geom("table").id
    cube = env.model.geom("cube_geom").id
    resting_z = (
        env.data.geom_xpos[table, 2] + env.model.geom_size[table, 2] + env.model.geom_size[cube, 1]
    )
    assert env.data.body("cube").xpos[2] == pytest.approx(resting_z, abs=0.003)
    assert env.model.joint("cube_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.neq == 0


def test_invalid_command_does_not_mutate_state(env):
    before = env.data.qpos.copy()
    targets = env.targets.copy()
    command = Command.hold()
    command.right = Pose(position=(4, 0, 1), quaternion=(1, 0, 0, 0))
    with pytest.raises(ValueError, match="workspace"):
        env.move(command)
    np.testing.assert_array_equal(before, env.data.qpos)
    assert env.targets == targets


def test_reject_nonfinite_bad_quaternion():
    with pytest.raises(ValueError):
        Pose(position=(float("nan"), 0, 0), quaternion=(1, 0, 0, 0))
    with pytest.raises(ValueError):
        Pose(position=(0, 0, 0), quaternion=(0, 0, 0, 0))
    assert "prefixItems" not in json.dumps(output_schema())
