"""Standard Inspect agent and 16-D Cartesian G1 embodiment."""

from dataclasses import replace

from inspect_robots.embodiment import EmbodimentBase
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.types import ActionChunk, StepResult

from .. import inspect_contract as contract
from ..env import G1Env
from ..policies.codex import AppServer
from ..policies.inspect_agent import InspectPolicy
from ..verification import resolve_task

SPACE = contract.SPACE
OBS = contract.info().observation_space
unpack = contract.decode


def pack(command, state):
    state = {**state, "hands": dict(state["hands"])}
    for side in ("left", "right"):
        pose = getattr(command, side)
        if pose is not None:
            state["hands"][side] = pose.model_dump()
    values = contract.vector(state)
    for side, index in (("left", 7), ("right", 15)):
        opening = getattr(command, f"{side}_open")
        if opening is not None:
            values[index] = opening
    return values


class G1Embodiment(EmbodimentBase):
    info = contract.info()

    def __init__(self, render=True, task="auto", hand="dex1"):
        self.env = G1Env(hand=hand)
        self.info = contract.info(hand=hand)
        self.render_enabled = render
        self.instruction = None
        self.task_mode = task
        self.task = "none"
        self.step_count = 0

    def _observe(self):
        return contract.observation(
            self.env.observe(),
            {"ego": self.env.render("ego")} if self.render_enabled else {},
            self.instruction,
            self.step_count,
        )

    def reset(self, scene, *, seed=None):
        if scene.setup is not None or scene.target is not None:
            raise ValueError("Only the built-in tabletop scene is supported")
        self.instruction = scene.instruction
        self.env.reset()
        self.step_count = 0
        self.task = resolve_task(self.task_mode, scene.instruction)
        return self._observe()

    def step(self, action):
        command = unpack(action.data)
        command.status = action.meta.get("status", "move")
        if command.status == "move":
            self.env.move(command, duration=1 / contract.CONTROL_HZ, linear=True)
            decision = {"stop": False, "outcome": "continue"}
        else:
            decision = self.env.execute(command, task=self.task)
        self.step_count += 1
        return StepResult(
            self._observe(),
            terminated=decision["stop"],
            termination_reason=decision["outcome"] if decision["stop"] else None,
            info={"motion": self.env.report, "decision": decision},
        )

    def close(self):
        self.env.close()


class InspectCodexPolicy(PolicyBase):
    info = PolicyInfo(
        name="codex_g1", action_space=SPACE, observation_space=OBS, control_hz=contract.CONTROL_HZ
    )

    def __init__(self, model="gpt-6-astra", effort="low", timeout=120):
        self.model, self.effort, self.timeout = model, effort, timeout
        self.server = None
        self.policy = None
        self.history = []
        self.embodiment_info = contract.info()

    def bind(self, embodiment_info):
        self.embodiment_info = embodiment_info

    def reset(self, scene):
        self.close()
        self.history = []
        self.server = AppServer(timeout=self.timeout)
        try:
            self.policy = InspectPolicy(self.server, self.model, self.effort)
            self.policy.agent.bind(self.embodiment_info)
        except BaseException:
            self.close()
            raise

    def act(self, observation):
        chunk = self.policy.act_observation(observation)
        actions = []
        for action in chunk.actions:
            meta = dict(action.meta)
            if meta.pop("request_stop", False):
                # Let the embodiment record private terminal evaluation before stopping.
                meta["status"] = meta.pop("stop_reason")
            actions.append(replace(action, meta=meta))
        return ActionChunk(actions, control_hz=chunk.control_hz, meta=chunk.meta)

    def transcript(self):
        return self.policy.transcript() if self.policy else self.history

    def on_trial_end(self, *args):
        self.close()

    def close(self):
        if self.policy:
            self.history = self.policy.transcript()
            self.policy.close()
            self.policy = None
        if self.server:
            self.server.close()
            self.server = None
