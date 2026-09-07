import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from g1_llm_policy.adapters.inspect_robots import G1Embodiment, pack, unpack
from g1_llm_policy.commands import Command


def test_inspect_roundtrip_and_physics():
    env = G1Embodiment(render=False)
    try:
        env.reset(Scene(id="reach", instruction="Reach forward"))
        action = pack(Command.hold(), env.env.observe())
        action[8] += 0.003
        command = unpack(action)
        assert command.right.position[0] == action[8]
        result = env.step(Action(action))
        assert result.info["motion"]["right"]["reached"]
        assert result.observation.state["eef_state"].shape == (16,)
        assert np.isfinite(result.observation.state["eef_state"]).all()
    finally:
        env.close()
