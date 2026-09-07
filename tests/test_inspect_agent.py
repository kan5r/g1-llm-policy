import json

import numpy as np
import pytest
from inspect_robots_agent.policy import _SYSTEM_TEMPLATE

from g1_llm_policy import inspect_contract as contract
from g1_llm_policy.env import G1Env
from g1_llm_policy.policies import inspect_agent


def test_standard_agent_observation_prompt_tools_and_execution(monkeypatch):
    requests = []
    env = G1Env()
    initial = env.observe()
    target = initial["hands"]["right"]["position"][0] + 0.03
    replies = iter(
        [
            {
                "name": "move_to",
                "arguments": json.dumps(
                    {"targets": {"right_x": 20}, "note": "Test invalid target"}
                ),
            },
            {
                "name": "move_to",
                "arguments": json.dumps({"targets": {"right_x": target}, "note": "Move forward"}),
            },
            {"name": "done", "arguments": json.dumps({"summary": "Finished", "hindsight": "none"})},
        ]
    )

    class Session:
        def __init__(self, server, model, effort, *, instructions):
            self.instructions = instructions

        def generate(self, inputs, schema):
            requests.append((self.instructions, inputs, schema))
            return next(replies)

    monkeypatch.setattr(inspect_agent, "CodexPolicy", Session)
    policy = inspect_agent.InspectPolicy(None)
    try:
        # Privileged and custom diagnostics cannot reach upstream observation formatting.
        initial["task_feedback"] = {"success": True, "secret": "PRIVILEGED_SENTINEL"}
        initial["last_motion"] = {"secret": "DIAGNOSTIC_SENTINEL"}
        cmd = policy.act("Move forward", initial, {"ego": np.zeros((8, 8, 3), dtype=np.uint8)})
        assert len(requests) == 2  # upstream rejects the out-of-bounds tool call and retries
        assert _SYSTEM_TEMPLATE.format(name="g1_mujoco", budget=100) in requests[0][0]
        sent = json.dumps(requests)
        assert "PRIVILEGED_SENTINEL" not in sent
        assert "DIAGNOSTIC_SENTINEL" not in sent
        assert "state[eef_state]" in sent
        assert "outside" in json.dumps(requests[1][1])
        assert len(cmd._waypoints) == 3
        assert cmd._control_hz == 10
        start = env.data.time
        env.execute(cmd)
        assert env.data.time - start == pytest.approx(0.3)
        assert env.pose("right").position[0] > initial["hands"]["right"]["position"][0]
        stop = policy.act("Move forward", env.observe(), {})
        assert stop.status == "done"
        assert env.execute(stop, task="pickup")["stop"]
    finally:
        policy.close()
        env.close()


def test_contract_measured_opening_and_rotation_roundtrip():
    env = G1Env()
    try:
        state = env.observe()
        state["commanded_openings"] = {"left": 0, "right": 0}
        values = contract.vector(state)
        assert values[7] > 0.99 and values[15] > 0.99
        decoded = contract.decode(values)
        for side in ("left", "right"):
            np.testing.assert_allclose(getattr(decoded, side).position, env.pose(side).position)
            assert (
                abs(np.dot(getattr(decoded, side).quaternion, env.pose(side).quaternion)) > 0.999999
            )
        assert set(contract.observation(state, {}, "reach").state) == {"eef_state"}
    finally:
        env.close()


def test_standard_on_demand_capture(monkeypatch):
    requests = []
    replies = iter(
        [
            {"name": "take_pic", "arguments": json.dumps({"note": "Inspect the scene"})},
            {
                "name": "give_up",
                "arguments": json.dumps({"reason": "End test", "hindsight": "none"}),
            },
        ]
    )

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, inputs, schema):
            requests.append(inputs)
            return next(replies)

    monkeypatch.setattr(inspect_agent, "CodexPolicy", Session)
    env = G1Env()
    policy = inspect_agent.InspectPolicy(None, images="on_demand")
    try:
        command = policy.act("Observe", env.observe(), {"ego": np.zeros((8, 8, 3), dtype=np.uint8)})
        assert command.status == "give_up"
        assert not any(p["type"] == "image" for p in requests[0])
        assert any(p["type"] == "image" for p in requests[1])
    finally:
        policy.close()
        env.close()


def test_quaternion_shortest_path_and_invalid_targets():
    from inspect_robots_agent._llm import ToolCall

    from g1_llm_policy.quaternion_tools import angle, unit

    env = G1Env()
    policy = inspect_agent.InspectPolicy(None)
    try:
        obs = contract.observation(env.observe(), {}, "Rotate")
        values = obs.state["eef_state"].copy()
        values[11:15] = [np.cos(np.deg2rad(85)), 0, 0, np.sin(np.deg2rad(85))]
        obs = __import__("dataclasses").replace(obs, state={"eef_state": values})
        toolset = policy.agent._toolset

        def call(q):
            args = {
                "targets": dict(zip(("right_qw", "right_qx", "right_qy", "right_qz"), q)),
                "note": "Rotate",
            }
            return toolset.execute(ToolCall("test", "move_to", json.dumps(args)), obs)

        result = call([np.cos(np.deg2rad(-85)), 0, 0, np.sin(np.deg2rad(-85))])
        assert result.error is None
        previous = values[11:15]
        total = 0
        for action in result.chunk.actions:
            q = action.data[11:15]
            assert np.linalg.norm(q) == pytest.approx(1)
            step = angle(unit(previous), unit(q))
            assert step <= toolset.angular_step + 1e-8
            total += step
            previous = q
        assert np.rad2deg(total) == pytest.approx(20)
        opposite = call(-values[11:15])
        assert len(opposite.chunk.actions) == 1
        assert angle(values[11:15], opposite.chunk.actions[0].data[11:15]) < 1e-7
        assert call([0, 0, 0, 0]).error
        assert call([1]).error
        assert "rot6d" not in json.dumps(toolset.schemas())
        assert "right_qw" in json.dumps(toolset.schemas())
    finally:
        policy.close()
        env.close()
