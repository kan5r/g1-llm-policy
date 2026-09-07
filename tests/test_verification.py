import json
from pathlib import Path

from inspect_robots.scene import Scene
from inspect_robots.types import Action

from g1_llm_policy.adapters.inspect_robots import G1Embodiment
from g1_llm_policy.commands import Command
from g1_llm_policy.env import G1Env
from g1_llm_policy.verification import PickupMonitor


def test_false_done_ends_with_private_failure():
    env = G1Env(object_shape="box")
    # Reproduce the historical scene, independently of the current layout.
    env.model.geom_pos[env.model.geom("table").id] = [0.6, 0, 0.65]
    env.model.body_pos[env.model.body("cube").id] = [0.47, -0.16, 0.715]
    env.reset()
    try:
        for raw in json.loads((Path(__file__).parent / "fixtures/false_pickup.json").read_text()):
            result = env.execute(Command.model_validate(raw), task="pickup")
        assert result["stop"]
        assert result["outcome"] == "failed"
        assert result["verification"]["clearance_m"] < 0.001
        assert result["verification"]["contact_regions"] == {"left": [], "right": []}
        assert "task_feedback" not in env.observe()
    finally:
        env.close()


def test_lift_requires_continuous_contact_not_throw_or_history():
    monitor = PickupMonitor()
    contact = {"left": [], "right": ["thumb", "index"]}
    for _ in range(60):
        monitor.update(0.04, 0.01, contact, 0.01)
    assert monitor.report()["success"]
    monitor.update(0.04, 0.01, {"left": [], "right": []}, 0.01)
    assert not monitor.report()["success"]
    for _ in range(60):
        monitor.update(0.04, 1.0, contact, 0.01)
    assert not monitor.report()["success"]
    for _ in range(60):
        monitor.update(0.0, 0.01, contact, 0.01)
    assert not monitor.report()["success"]


def test_inspect_done_ends_without_feedback_leak():
    env = G1Embodiment(render=False)
    try:
        obs = env.reset(Scene(id="pickup", instruction="pick up the red block"))
        action = Action(obs.state["eef_state"], meta={"status": "done"})
        result = env.step(action)
        assert result.terminated
        assert result.termination_reason == "failed"
        assert "task_feedback" not in obs.extra
        assert "task_feedback" not in result.observation.extra
    finally:
        env.close()


def test_nonpickup_done_unchanged():
    env = G1Env()
    try:
        command = Command.hold().model_copy(update={"status": "done"})
        assert env.execute(command, task="none")["stop"]
    finally:
        env.close()


def test_cli_does_not_coach_after_false_done(monkeypatch, tmp_path):
    import sys
    from contextlib import nullcontext

    import numpy as np

    from g1_llm_policy import cli

    seen = []

    class Policy:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def transcript(self):
            return []

        def act(self, instruction, state, images):
            seen.append(state)
            return Command.hold().model_copy(
                update={"status": "done" if len(seen) == 1 else "give_up"}
            )

    monkeypatch.setattr(cli, "AppServer", lambda **kwargs: nullcontext(None))
    monkeypatch.setattr(cli, "CodexPolicy", Policy)
    monkeypatch.setattr(G1Env, "render", lambda *args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "run",
            "--no-video",
            "--max-steps",
            "2",
            "--task",
            "pickup",
            "--output",
            str(tmp_path),
        ],
    )
    cli.main()
    assert len(seen) == 1
    assert "task_feedback" not in seen[0]
    rows = [json.loads(line) for line in (tmp_path / "episode.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["decision"]["stop"]
    assert rows[0]["decision"]["outcome"] == "failed"
    assert rows[0]["physics"]["success"] is False
