"""CPU physics + differential IK. Physics never copies the IK configuration."""

import time
from collections.abc import Callable

import mink
import mujoco
import numpy as np

from .commands import Command, Pose
from .model import build_model
from .verification import PickupMonitor


class G1Env:
    def __init__(self, object_shape="box", hand="dex1"):
        self.hand = hand
        self.model = build_model(object_shape=object_shape, hand=hand)
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.tasks = {
            side: mink.FrameTask(
                f"{side}_palm", "site", position_cost=1.0, orientation_cost=0.3, lm_damping=1.0
            )
            for side in ("left", "right")
        }
        self.posture = mink.PostureTask(self.model, cost=0.0001)
        self.robot_joints = [
            self.model.joint(i)
            for i in range(self.model.njnt)
            if self.model.joint(i).name != "cube_free"
        ]
        velocity_limits = {
            j.name: 0.2 if "dex1_finger" in j.name else 1.5 for j in self.robot_joints
        }
        hand_geoms = []
        for i in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[i]).name
            if (
                "hand" in body or "wrist_yaw" in body or "dex1" in body
            ) and self.model.geom_contype[i]:
                hand_geoms.append(i)
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, velocity_limits),
            mink.CollisionAvoidanceLimit(
                self.model,
                geom_pairs=[(hand_geoms, [self.model.geom("table").id])],
                minimum_distance_from_collisions=0.003,
                collision_detection_distance=0.06,
            ),
        ]
        self.renderer = None
        self.reset()

    def reset(self):
        self.pickup = PickupMonitor()
        self.approval_store = {}
        self.approval_feedback = []
        mujoco.mj_resetData(self.model, self.data)
        # In this MJCF, elbow q=0 already gives an approximately 90-degree bend.
        # All arm joints therefore start at zero for the requested neutral pose.
        if self.hand == "dex1":
            from .dex1 import OPEN

            for joint in self.robot_joints:
                if "dex1_finger" in joint.name:
                    self.data.qpos[joint.qposadr[0]] = OPEN
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture.set_target_from_configuration(self.configuration)
        self.targets = {s: self.pose(s) for s in self.tasks}
        self.openings = {"left": 1.0, "right": 1.0}
        self.report = {}
        self._set_controls(self.data.qpos)
        self._physics(100)
        self.targets = {s: self.pose(s) for s in self.tasks}
        return self.observe()

    def pose(self, side):
        site = self.data.site(f"{side}_palm")
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, site.xmat)
        return Pose(position=tuple(site.xpos), quaternion=tuple(q))

    def _set_controls(self, qpos):
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            joint = self.model.joint(joint_id)
            target = qpos[joint.qposadr[0]]
            if "dex1_finger" in joint.name:
                from .dex1 import CLOSED, OPEN

                side = "left" if joint.name.startswith("left") else "right"
                target = CLOSED + self.openings[side] * (OPEN - CLOSED)
            elif "hand" in joint.name:
                side = "left" if joint.name.startswith("left") else "right"
                close = 1 - self.openings[side]
                # Symmetric fixed power-grasp reference, kept inside the model ranges.
                sign = 1 if side == "left" else -1
                if "thumb_0" in joint.name:
                    target = 0
                elif "thumb_1" in joint.name:
                    target = sign * (1.0 - 0.35 * close)
                elif "thumb_2" in joint.name:
                    target = sign * close * 0.65
                else:
                    target = -sign * close * 0.85
            lo, hi = self.model.actuator_ctrlrange[i]
            self.data.ctrl[i] = np.clip(target, lo, hi)

    def _physics(self, steps):
        for _ in range(steps):
            # Gravity/bias compensation on actuated joints only; cube remains free.
            self.data.qfrc_applied[:] = 0
            for joint in self.robot_joints:
                adr = joint.dofadr[0]
                self.data.qfrc_applied[adr] = self.data.qfrc_bias[adr]
            mujoco.mj_step(self.model, self.data)
            self.pickup.sample(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)):
            raise RuntimeError("Non-finite physical state")

    def move(
        self,
        command: Command,
        duration=2.0,
        callback: Callable | None = None,
        realtime=False,
        linear=False,
    ):
        if command.status != "move":
            return self.observe()
        if not 0 < duration <= 10:
            raise ValueError("duration must be in (0,10] seconds")
        starts = {s: self.pose(s) for s in self.tasks}
        requested = self.targets.copy()
        for side in self.tasks:
            pose = getattr(command, side)
            if pose is not None:
                p = np.asarray(pose.position)
                if not (np.all(p >= [-0.25, -0.85, 0.4]) and np.all(p <= [0.85, 0.85, 1.65])):
                    raise ValueError("Target outside world-frame workspace")
                requested[side] = pose
        # Validate all targets before changing the held command.
        self.targets = requested
        for side in self.tasks:
            opening = getattr(command, f"{side}_open")
            if opening is not None:
                self.openings[side] = opening
        dt = 0.01
        ticks = max(1, round(duration / dt))
        start_time = time.monotonic()
        self.configuration.update(self.data.qpos)
        for tick in range(ticks):
            alpha = min(1, (tick + 1) / max(1, ticks if linear else ticks * 0.7))
            for side, task in self.tasks.items():
                a = mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.array(starts[side].quaternion)), np.array(starts[side].position)
                )
                b = mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.array(self.targets[side].quaternion)),
                    np.array(self.targets[side].position),
                )
                task.set_target(a @ mink.SE3.exp(alpha * (a.inverse() @ b).log()))
            vel = mink.solve_ik(
                self.configuration,
                [*self.tasks.values(), self.posture],
                dt,
                "daqp",
                damping=1e-3,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, dt)
            self._set_controls(self.configuration.q)
            self._physics(5)
            if callback:
                callback(self)
            if realtime:
                time.sleep(max(0, start_time + (tick + 1) * dt - time.monotonic()))
        self.report = {}
        for side in self.tasks:
            actual = self.pose(side)
            pos_error = float(
                np.linalg.norm(np.array(actual.position) - self.targets[side].position)
            )
            angle_error = float(
                np.linalg.norm(
                    (
                        mink.SO3(np.array(actual.quaternion)).inverse()
                        @ mink.SO3(np.array(self.targets[side].quaternion))
                    ).log()
                )
            )
            ik_pose = self.configuration.get_transform_frame_to_world(f"{side}_palm", "site")
            ik_error = float(np.linalg.norm(ik_pose.translation() - self.targets[side].position))
            tracking_error = float(np.linalg.norm(ik_pose.translation() - actual.position))
            self.report[side] = {
                "ik_position_error_m": ik_error,
                "tracking_position_error_m": tracking_error,
                "position_error_m": pos_error,
                "orientation_error_rad": angle_error,
                "reached": pos_error < 0.02 and angle_error < 0.15,
            }
        return self.observe()

    def play_chunk(self, command, *, callback=None, realtime=False):
        from types import SimpleNamespace

        from inspect_robots.controller import DefaultController
        from inspect_robots.types import Action

        from .inspect_contract import SPACE, decode, vector

        controller = DefaultController()
        from .quaternion_tools import QuaternionApprover

        approver = QuaternionApprover(SPACE)
        # Seed the first delta check from measured proprioception.
        if not self.approval_store:
            approver.review(Action(vector(self.observe())), self.approval_store)
        fixed_policy = SimpleNamespace(act=lambda obs: command._action_chunk)
        store = {}
        self.approval_feedback = []
        for tick in range(len(command._action_chunk.actions)):
            action = controller.next_action(fixed_policy, None, tick, store)
            action = approver.review(action, self.approval_store)
            flags = [key for key in ("clamped", "delta_clamped") if action.meta.get(key)]
            if flags:
                self.approval_feedback.append({"detail": ", ".join(flags)})
            waypoint = decode(action.data, command.note)
            self.move(
                waypoint,
                duration=1 / command._control_hz,
                callback=callback,
                realtime=realtime,
                linear=True,
            )

    def execute(self, command, *, task="none", callback=None, realtime=False):
        """Execute a command; evaluate pickup privately without coaching the policy."""
        if command.status == "move":
            try:
                if command._action_chunk is not None:
                    self.play_chunk(command, callback=callback, realtime=realtime)
                else:
                    self.move(command, callback=callback, realtime=realtime)
            except ValueError as exc:
                self.report = {"rejected": str(exc)}
        elif command.status == "done" and task == "pickup":
            # Keep current actuator targets; verify persistence instead of a single frame.
            started = time.monotonic()
            for tick in range(75):
                self._physics(5)
                if callback:
                    callback(self)
                if realtime:
                    time.sleep(max(0, started + (tick + 1) * 0.01 - time.monotonic()))
        verification = self.pickup.report() if task == "pickup" else None
        stop = command.status in ("done", "give_up")
        outcome = command.status if stop else "continue"
        if command.status == "done" and verification is not None:
            outcome = "success" if verification["success"] else "failed"
        return {"stop": stop, "outcome": outcome, "verification": verification}

    def observe(self):
        return {
            "gripper_model": self.hand,
            "tcp_wrist_local_m": [0.155 if self.hand == "dex1" else 0.08, 0, 0],
            "frame": "world",
            "position_unit": "m",
            "quaternion_order": "wxyz",
            "sim_time_s": float(self.data.time),
            "hands": {s: self.pose(s).model_dump() for s in self.tasks},
            "commanded_openings": self.openings.copy(),
            "targets": {s: p.model_dump() for s, p in self.targets.items()},
            "joints": {j.name: float(self.data.qpos[j.qposadr[0]]) for j in self.robot_joints},
            "last_motion": self.report.copy(),
            "approvals": list(self.approval_feedback),
        }

    def render(self, camera="overview"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
